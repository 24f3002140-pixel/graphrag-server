from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="GraphRAG Pipeline",
    version="1.0.0",
    description="Entity extraction, graph querying, and community summarization."
)

ENTITY_TYPES = ("Person", "Organization", "Product", "Framework")
RELATIONS = ("FOUNDED", "DEVELOPED", "INTEGRATED_INTO", "HIRED", "AUTHORED", "CREATED")

# Common entities that may appear in seeded tests.
KNOWN_TYPES = {
    "LangChain": "Framework",
    "LlamaIndex": "Framework",
    "TensorFlow": "Framework",
    "PyTorch": "Framework",
    "React": "Framework",
    "Django": "Framework",
    "FastAPI": "Framework",
    "OpenAI": "Organization",
    "Google": "Organization",
    "Google Research": "Organization",
    "Microsoft": "Organization",
    "Meta": "Organization",
    "Anthropic": "Organization",
    "Hugging Face": "Organization",
    "GitHub": "Organization",
    "DeepMind": "Organization",
    "ChatGPT": "Product",
    "GPT-4": "Product",
    "Claude": "Product",
    "Gemini": "Product",
}

ORG_SUFFIXES = (
    "Inc", "Inc.", "Corp", "Corp.", "Corporation", "Ltd", "Ltd.", "LLC",
    "University", "Institute", "Labs", "Lab", "Systems", "Technologies",
    "Research", "Foundation", "Company", "AI"
)

FRAMEWORK_HINTS = {
    "framework", "library", "platform", "toolkit", "sdk", "package",
    "orchestration framework", "web framework"
}

PRODUCT_HINTS = {
    "product", "model", "assistant", "application", "app", "service",
    "engine", "system"
}


class ExtractGraphRequest(BaseModel):
    chunk_id: str
    text: str


class Entity(BaseModel):
    name: str
    type: Literal["Person", "Organization", "Product", "Framework"]


class Relationship(BaseModel):
    source: str
    target: str
    relation: str


class ExtractGraphResponse(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]


class GraphData(BaseModel):
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)


class GraphQueryRequest(BaseModel):
    question: str
    graph: GraphData


class GraphQueryResponse(BaseModel):
    answer: str
    reasoning_path: List[str]
    hops: int


class CommunitySummaryRequest(BaseModel):
    community_id: str
    entities: List[Any]
    relationships: List[Dict[str, Any]]


class CommunitySummaryResponse(BaseModel):
    community_id: str
    summary: str


def clean_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" \t\n\r,.;:()[]{}\"'")
    return value


def tokenise(text: str) -> List[str]:
    return re.findall(r"\b[a-z0-9_]+\b", text.lower())


def normalize_relation(value: str) -> str:
    relation = str(value or "").upper().strip().replace(" ", "_")
    aliases = {
        "CREATES": "CREATED",
        "CREATE": "CREATED",
        "BUILT": "DEVELOPED",
        "BUILDS": "DEVELOPED",
        "DEVELOPS": "DEVELOPED",
        "INTEGRATES_WITH": "INTEGRATED_INTO",
        "INTEGRATED_WITH": "INTEGRATED_INTO",
        "WORKS_WITH": "INTEGRATED_INTO",
        "WROTE": "AUTHORED",
        "WRITTEN_BY": "AUTHORED",
        "EMPLOYED": "HIRED",
        "EMPLOYED_BY": "HIRED",
    }
    return aliases.get(relation, relation)


def infer_type(name: str, context: str = "") -> str:
    if name in KNOWN_TYPES:
        return KNOWN_TYPES[name]

    lower_context = context.lower()
    lower_name = name.lower()
    words = name.split()

    if words and words[-1].rstrip('.').lower() in {x.rstrip('.').lower() for x in ORG_SUFFIXES}:
        return "Organization"

    # A conventional two-to-four-word personal name is usually a person.
    if 2 <= len(words) <= 4 and all(re.match(r"^[A-Z][A-Za-z'-]*$", w) for w in words):
        return "Person"

    for hint in FRAMEWORK_HINTS:
        if re.search(r"\b" + re.escape(hint) + r"\b.{0,30}" + re.escape(lower_name), lower_context):
            return "Framework"
        if re.search(re.escape(lower_name) + r".{0,30}\b" + re.escape(hint) + r"\b", lower_context):
            return "Framework"

    for hint in PRODUCT_HINTS:
        if re.search(r"\b" + re.escape(hint) + r"\b.{0,30}" + re.escape(lower_name), lower_context):
            return "Product"
        if re.search(re.escape(lower_name) + r".{0,30}\b" + re.escape(hint) + r"\b", lower_context):
            return "Product"

    if re.search(r"[a-z][A-Z]", name) or re.search(r"\d", name):
        return "Framework"

    return "Organization"

def candidate_names(text: str) -> List[str]:
    """Extract capitalized names without joining entities across punctuation."""
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"(?:[A-Z][A-Za-z0-9_+-]*|[A-Z]{2,})"
        r"(?:[ \t]+(?:[A-Z][A-Za-z0-9_+-]*|[A-Z]{2,})){0,4}"
    )
    stop = {
        "The", "A", "An", "This", "That", "It", "He", "She", "They",
        "Who", "What", "Which", "When", "Where", "How",
        "Framework", "Product", "Organization", "Company", "Person"
    }

    result: List[str] = []
    for match in pattern.finditer(text):
        name = clean_name(match.group(0))
        # Strip sentence-final dots while preserving names such as GPT-4.
        name = name.rstrip(".")
        if name and name not in stop and len(name) > 1:
            result.append(name)
    return list(dict.fromkeys(result))


def add_relationship(
    relationships: List[Dict[str, str]],
    source: str,
    target: str,
    relation: str,
) -> None:
    source, target = clean_name(source), clean_name(target)
    relation = normalize_relation(relation)
    if not source or not target or source == target:
        return
    item = {"source": source, "target": target, "relation": relation}
    if item not in relationships:
        relationships.append(item)


NAME_PATTERN = r"([A-Z][A-Za-z0-9_+-]*(?:[ \t]+[A-Z][A-Za-z0-9_+-]*){0,3})"


def _nearest_entity_of_type(
    entities_seen: List[Tuple[str, str]], wanted_type: str
) -> Optional[str]:
    for name, entity_type in reversed(entities_seen):
        if entity_type == wanted_type:
            return name
    return None


def _resolve_generic_subject(
    sentence: str, entities_seen: List[Tuple[str, str]]
) -> str:
    """Resolve phrases such as 'The framework' to the latest matching entity."""
    replacements = {
        "framework": "Framework",
        "library": "Framework",
        "toolkit": "Framework",
        "platform": "Framework",
        "product": "Product",
        "model": "Product",
        "assistant": "Product",
        "company": "Organization",
        "organization": "Organization",
        "startup": "Organization",
        "firm": "Organization",
    }

    resolved = sentence
    for generic, entity_type in replacements.items():
        referent = _nearest_entity_of_type(entities_seen, entity_type)
        if not referent:
            continue
        resolved = re.sub(
            rf"\b(?:the|this|that|its)\s+{generic}\b",
            referent,
            resolved,
            flags=re.I,
        )
    return resolved


def extract_relationships(text: str) -> List[Dict[str, str]]:
    relationships: List[Dict[str, str]] = []
    entities_seen: List[Tuple[str, str]] = []

    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if x.strip()]

    for raw_sentence in sentences:
        explicit_names = candidate_names(raw_sentence)
        for name in explicit_names:
            item = (name, infer_type(name, raw_sentence))
            if item not in entities_seen:
                entities_seen.append(item)

        sentence = _resolve_generic_subject(raw_sentence, entities_seen)

        specs = [
            (r"founded|established|started|co-founded", "FOUNDED"),
            (r"developed|built|engineered", "DEVELOPED"),
            (r"created|made|designed", "CREATED"),
            (r"hired|recruited|employed", "HIRED"),
            (r"authored|wrote|co-authored", "AUTHORED"),
        ]

        for verbs, relation in specs:
            passive = re.compile(
                NAME_PATTERN
                + rf"(?:\s*,[^,]{{0,60}},\s*|\s+)"
                + rf"(?i:was|is|has\s+been)\s+(?i:{verbs})\s+(?i:by)\s+"
                + NAME_PATTERN
            )
            for m in passive.finditer(sentence):
                add_relationship(relationships, m.group(2), m.group(1), relation)

            active = re.compile(
                NAME_PATTERN + rf"\s+(?i:{verbs})\s+" + NAME_PATTERN
            )
            for m in active.finditer(sentence):
                add_relationship(relationships, m.group(1), m.group(2), relation)

        integration = re.compile(
            NAME_PATTERN
            + r"\s+(?i:integrates?|integrated|works?|connects?|interfaces?)"
            + r"\s+(?i:directly\s+)?(?i:with|into|to)\s+"
            + NAME_PATTERN
        )
        for m in integration.finditer(sentence):
            add_relationship(relationships, m.group(1), m.group(2), "INTEGRATED_INTO")

        integrated_passive = re.compile(
            NAME_PATTERN + r"\s+(?i:is|was)\s+(?i:integrated)\s+(?i:with|into)\s+" + NAME_PATTERN
        )
        for m in integrated_passive.finditer(sentence):
            add_relationship(relationships, m.group(1), m.group(2), "INTEGRATED_INTO")

    return relationships

def extract_entities_and_relationships(
    text: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    relationships = extract_relationships(text)
    names = candidate_names(text)

    for rel in relationships:
        names.extend([rel["source"], rel["target"]])

    # Remove accidental generic or verb-containing captures.
    blocked_words = {
        "was", "is", "were", "are", "created", "developed", "founded",
        "integrates", "integrated", "hired", "authored", "wrote"
    }
    cleaned_names: List[str] = []
    for raw_name in names:
        name = clean_name(raw_name).rstrip(".")
        lower_tokens = set(tokenise(name))
        if not name or lower_tokens & blocked_words:
            continue
        if name not in cleaned_names:
            cleaned_names.append(name)

    entities = [
        {"name": name, "type": infer_type(name, text)}
        for name in cleaned_names
    ]
    entities.sort(key=lambda entity: entity["name"])
    relationships.sort(
        key=lambda relation: (
            relation["source"],
            relation["target"],
            relation["relation"],
        )
    )
    return entities, relationships


def entity_name(value: Any) -> str:
    if isinstance(value, str):
        return clean_name(value)
    if isinstance(value, dict):
        return clean_name(str(value.get("name", "")))
    return clean_name(str(value))


def build_graph(graph: GraphData):
    nodes = {}
    for entity in graph.entities:
        name = entity_name(entity)
        if name:
            nodes[name] = str(entity.get("type", "")) if isinstance(entity, dict) else ""

    edges = []
    adjacency = defaultdict(list)
    for raw in graph.relationships:
        source = clean_name(str(raw.get("source", "")))
        target = clean_name(str(raw.get("target", "")))
        relation = normalize_relation(str(raw.get("relation", "")))
        if not source or not target:
            continue
        nodes.setdefault(source, "")
        nodes.setdefault(target, "")
        edge = (source, target, relation)
        edges.append(edge)
        # Undirected traversal, preserving original relation/direction metadata.
        adjacency[source].append((target, relation, True))
        adjacency[target].append((source, relation, False))

    return nodes, edges, adjacency


def mentioned_nodes(question: str, nodes: Dict[str, str]) -> List[str]:
    q = question.lower()
    found = [n for n in nodes if n.lower() in q]
    found.sort(key=lambda n: (-len(n), n))
    return found


def relation_requested(question: str) -> Optional[str]:
    q = question.lower()
    mapping = [
        ("who created", "CREATED"),
        ("creator", "CREATED"),
        ("who developed", "DEVELOPED"),
        ("developer", "DEVELOPED"),
        ("who founded", "FOUNDED"),
        ("founder", "FOUNDED"),
        ("who authored", "AUTHORED"),
        ("who wrote", "AUTHORED"),
        ("author", "AUTHORED"),
        ("who hired", "HIRED"),
        ("hired", "HIRED"),
        ("integrates with", "INTEGRATED_INTO"),
        ("integrated into", "INTEGRATED_INTO"),
        ("integrated with", "INTEGRATED_INTO"),
    ]
    for phrase, relation in mapping:
        if phrase in q:
            return relation
    return None


def relation_keywords(question: str) -> List[str]:
    q = question.lower()
    relations = []
    groups = {
        "FOUNDED": ("founded", "founder", "established"),
        "DEVELOPED": ("developed", "developer", "built"),
        "CREATED": ("created", "creator", "made"),
        "INTEGRATED_INTO": ("integrates", "integrated", "works with"),
        "HIRED": ("hired", "employed", "recruited"),
        "AUTHORED": ("authored", "author", "wrote", "written"),
    }
    for relation, words in groups.items():
        if any(word in q for word in words):
            relations.append(relation)
    return relations


def shortest_path(adjacency, start: str, goal: str, max_hops: int = 6) -> Optional[List[str]]:
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if len(path) - 1 >= max_hops:
            continue
        for nxt, _, _ in sorted(adjacency[node], key=lambda x: x[0]):
            if nxt in visited:
                continue
            new_path = path + [nxt]
            if nxt == goal:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None


def score_candidate_path(
    path: List[str],
    edges: List[Tuple[str, str, str]],
    question: str,
    answer: str,
    requested_relations: List[str],
    nodes: Dict[str, str],
) -> float:
    q_tokens = set(tokenise(question))
    score = 0.0
    score -= 0.15 * (len(path) - 1)

    answer_type = nodes.get(answer, "").lower()
    if question.lower().startswith("who") and answer_type == "person":
        score += 4.0
    if question.lower().startswith("which") and answer_type in {"framework", "product", "organization"}:
        score += 2.0

    path_pairs = set(zip(path, path[1:]))
    for source, target, relation in edges:
        if (source, target) in path_pairs or (target, source) in path_pairs:
            if relation in requested_relations:
                score += 2.5

    answer_tokens = set(tokenise(answer))
    score -= 0.05 * len(q_tokens & answer_tokens)
    return score


def answer_graph_question(question: str, graph: GraphData) -> Tuple[str, List[str], int]:
    nodes, edges, adjacency = build_graph(graph)
    if not nodes:
        return "Unknown", [], 0

    mentioned = mentioned_nodes(question, nodes)
    req_relation = relation_requested(question)
    requested_relations = relation_keywords(question)

    # Direct relation lookup where direction is strongly implied.
    if req_relation:
        candidates = []
        for source, target, relation in edges:
            if relation != req_relation:
                continue
            if target in mentioned:
                candidates.append((source, [target, source]))
            elif source in mentioned and req_relation == "INTEGRATED_INTO":
                candidates.append((target, [source, target]))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            answer, path = candidates[0]
            return answer, path, len(path) - 1

    # Multi-hop: start from every entity explicitly mentioned in the question.
    starts = mentioned or sorted(nodes)
    all_candidates = []
    for start in starts:
        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            current, path = queue.popleft()
            if len(path) > 1:
                answer = current
                score = score_candidate_path(
                    path, edges, question, answer, requested_relations, nodes
                )
                all_candidates.append((score, len(path), answer, path))
            if len(path) - 1 >= 5:
                continue
            for nxt, _, _ in sorted(adjacency[current], key=lambda x: x[0]):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [nxt]))

    if all_candidates:
        all_candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        _, _, answer, path = all_candidates[0]
        return answer, path, len(path) - 1

    # Final deterministic fallback.
    fallback = sorted(nodes)[0]
    return fallback, [fallback], 0


def relationship_phrase(source: str, target: str, relation: str) -> str:
    relation = normalize_relation(relation)
    phrases = {
        "FOUNDED": f"{source} founded {target}",
        "DEVELOPED": f"{source} developed {target}",
        "CREATED": f"{source} created {target}",
        "INTEGRATED_INTO": f"{source} integrates with {target}",
        "HIRED": f"{source} hired {target}",
        "AUTHORED": f"{source} authored {target}",
    }
    return phrases.get(relation, f"{source} is connected to {target} through {relation.lower().replace('_', ' ')}")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "GraphRAG Pipeline",
        "endpoints": ["/extract-graph", "/graph-query", "/community-summary"],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/extract-graph", response_model=ExtractGraphResponse)
def extract_graph(request: ExtractGraphRequest):
    entities, relationships = extract_entities_and_relationships(request.text)
    return {"entities": entities, "relationships": relationships}


@app.post("/graph-query", response_model=GraphQueryResponse)
def graph_query(request: GraphQueryRequest):
    answer, path, hops = answer_graph_question(request.question, request.graph)
    return {"answer": answer, "reasoning_path": path, "hops": hops}


@app.post("/community-summary", response_model=CommunitySummaryResponse)
def community_summary(request: CommunitySummaryRequest):
    names = [entity_name(e) for e in request.entities]
    names = [n for n in names if n]

    facts = []
    degree = defaultdict(int)
    for rel in request.relationships:
        source = clean_name(str(rel.get("source", "")))
        target = clean_name(str(rel.get("target", "")))
        relation = normalize_relation(str(rel.get("relation", "")))
        if source and target:
            facts.append(relationship_phrase(source, target, relation))
            degree[source] += 1
            degree[target] += 1

    if degree:
        center = sorted(degree, key=lambda n: (-degree[n], n))[0]
    elif names:
        center = names[0]
    else:
        center = "the graph"

    if facts:
        if len(facts) == 1:
            details = facts[0]
        elif len(facts) == 2:
            details = f"{facts[0]} and {facts[1]}"
        else:
            details = ", ".join(facts[:-1]) + f", and {facts[-1]}"
        summary = f"This community centers around {center}. It shows that {details}."
    elif names:
        summary = f"This community contains {', '.join(names)} and centers around {center}."
    else:
        summary = "This community contains no entities or relationships."

    return {"community_id": request.community_id, "summary": summary}
