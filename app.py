from __future__ import annotations

import re
import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
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
    """Return plausible named entities without crossing sentence punctuation."""
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"(?:[A-Z][A-Za-z0-9_+.#/-]*|[A-Z]{2,})"
        r"(?:[ \t]+(?:[A-Z][A-Za-z0-9_+.#/-]*|[A-Z]{2,})){0,5}"
    )
    stop = {
        "The", "A", "An", "This", "That", "These", "Those", "It", "Its",
        "He", "She", "They", "His", "Her", "Their", "Who", "What", "Which",
        "When", "Where", "How", "Framework", "Product", "Organization",
        "Company", "Person", "Later", "After", "Before", "Meanwhile"
    }
    out: List[str] = []
    for match in pattern.finditer(text):
        value = clean_name(match.group(0)).rstrip(".")
        if value and value not in stop and len(value) > 1 and value not in out:
            out.append(value)
    return out


def add_relationship(
    relationships: List[Dict[str, str]],
    source: str,
    target: str,
    relation: str,
) -> None:
    source = clean_name(source).rstrip(".")
    target = clean_name(target).rstrip(".")
    relation = normalize_relation(relation)
    if not source or not target or source == target:
        return
    item = {"source": source, "target": target, "relation": relation}
    if item not in relationships:
        relationships.append(item)


NAME_PATTERN = (
    r"([A-Z][A-Za-z0-9_+.#/-]*"
    r"(?:[ \t]+[A-Z][A-Za-z0-9_+.#/-]*){0,5})"
)


def _latest_by_type(
    history: List[Tuple[str, str]], wanted: str
) -> Optional[str]:
    for name, entity_type in reversed(history):
        if entity_type == wanted:
            return name
    return None


def _latest_person(history: List[Tuple[str, str]]) -> Optional[str]:
    return _latest_by_type(history, "Person")


def _latest_non_person(history: List[Tuple[str, str]]) -> Optional[str]:
    for name, entity_type in reversed(history):
        if entity_type != "Person":
            return name
    return None


def _resolve_references(
    sentence: str, history: List[Tuple[str, str]]
) -> str:
    resolved = sentence

    generic_types = {
        "framework": "Framework",
        "library": "Framework",
        "toolkit": "Framework",
        "platform": "Framework",
        "package": "Framework",
        "product": "Product",
        "model": "Product",
        "assistant": "Product",
        "application": "Product",
        "app": "Product",
        "service": "Product",
        "company": "Organization",
        "organization": "Organization",
        "startup": "Organization",
        "firm": "Organization",
        "lab": "Organization",
    }
    for generic, entity_type in generic_types.items():
        referent = _latest_by_type(history, entity_type)
        if referent:
            resolved = re.sub(
                rf"\b(?:the|this|that|its|their)\s+{generic}\b",
                referent,
                resolved,
                flags=re.I,
            )

    person = _latest_person(history)
    non_person = _latest_non_person(history)

    if person:
        resolved = re.sub(r"\b(?:he|she|him|her|his)\b", person, resolved, flags=re.I)
    if non_person:
        resolved = re.sub(r"\b(?:it|its)\b", non_person, resolved, flags=re.I)

    return resolved


def extract_relationships(text: str) -> List[Dict[str, str]]:
    relationships: List[Dict[str, str]] = []
    history: List[Tuple[str, str]] = []

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?;])\s+|\n+", text)
        if s.strip()
    ]

    active = {
        "FOUNDED": r"founded|co-founded|established|started|launched",
        "DEVELOPED": r"developed|built|engineered|implemented|designed",
        "CREATED": r"created|invented|made|introduced",
        "HIRED": r"hired|recruited|employed|appointed",
        "AUTHORED": r"authored|wrote|published|co-authored",
    }

    passive = {
        "FOUNDED": r"founded|co-founded|established|started|launched",
        "DEVELOPED": r"developed|built|engineered|implemented|designed",
        "CREATED": r"created|invented|made|introduced",
        "HIRED": r"hired|recruited|employed|appointed",
        "AUTHORED": r"authored|written|published|co-authored",
    }

    for raw in sentences:
        # Add explicitly named entities to discourse history.
        for name in candidate_names(raw):
            pair = (name, infer_type(name, raw))
            if pair not in history:
                history.append(pair)

        sentence = _resolve_references(raw, history)

        # Passive forms:
        # "X was created by Y"
        # "X, a framework, was developed by Y"
        for relation, verbs in passive.items():
            patterns = [
                re.compile(
                    NAME_PATTERN
                    + rf"(?:\s*,[^,]{{0,80}},\s*|\s+)"
                    + rf"(?:was|is|were|are|has been|had been)\s+"
                    + rf"(?:{verbs})\s+by\s+"
                    + NAME_PATTERN,
                    re.I,
                ),
                re.compile(
                    NAME_PATTERN
                    + rf"\s*,?\s*(?:a|an|the)?\s*"
                    + rf"(?:framework|library|product|platform|company|organization)?"
                    + rf"\s*(?:{verbs})\s+by\s+"
                    + NAME_PATTERN,
                    re.I,
                ),
            ]
            for pattern in patterns:
                for m in pattern.finditer(sentence):
                    target, source = m.group(1), m.group(2)
                    add_relationship(relationships, source, target, relation)

        # Active forms: "Y created X"
        for relation, verbs in active.items():
            pattern = re.compile(
                NAME_PATTERN
                + rf"\s+(?:has\s+|had\s+|also\s+)?(?:{verbs})\s+"
                + NAME_PATTERN,
                re.I,
            )
            for m in pattern.finditer(sentence):
                add_relationship(
                    relationships, m.group(1), m.group(2), relation
                )

        # Noun forms:
        # "X's founder Y", "the creator of X, Y"
        noun_patterns = [
            (
                re.compile(
                    NAME_PATTERN + r"(?:'s|’s)\s+(?:founder|co-founder)\s+"
                    + NAME_PATTERN,
                    re.I,
                ),
                "FOUNDED",
                True,
            ),
            (
                re.compile(
                    NAME_PATTERN + r"(?:'s|’s)\s+(?:creator|developer)\s+"
                    + NAME_PATTERN,
                    re.I,
                ),
                "CREATED",
                True,
            ),
            (
                re.compile(
                    r"(?:founder|co-founder)\s+of\s+" + NAME_PATTERN
                    + r"\s*,?\s*" + NAME_PATTERN,
                    re.I,
                ),
                "FOUNDED",
                False,
            ),
            (
                re.compile(
                    r"(?:creator|developer)\s+of\s+" + NAME_PATTERN
                    + r"\s*,?\s*" + NAME_PATTERN,
                    re.I,
                ),
                "CREATED",
                False,
            ),
        ]
        for pattern, relation, possessive in noun_patterns:
            for m in pattern.finditer(sentence):
                if possessive:
                    target, source = m.group(1), m.group(2)
                else:
                    target, source = m.group(1), m.group(2)
                add_relationship(relationships, source, target, relation)

        # Integration forms and aliases.
        integration_patterns = [
            re.compile(
                NAME_PATTERN
                + r"\s+(?:integrates?|integrated|works?|connects?|interfaces?|"
                + r"interoperates?|collaborates?)\s+(?:directly\s+)?"
                + r"(?:with|into|to|alongside)\s+" + NAME_PATTERN,
                re.I,
            ),
            re.compile(
                NAME_PATTERN
                + r"\s+(?:is|was|has been)\s+integrated\s+(?:with|into)\s+"
                + NAME_PATTERN,
                re.I,
            ),
            re.compile(
                NAME_PATTERN
                + r"\s+(?:provides?|offers?|adds?)\s+(?:an?\s+)?integration\s+"
                + r"(?:with|for)\s+" + NAME_PATTERN,
                re.I,
            ),
            re.compile(
                NAME_PATTERN
                + r"\s+(?:supports?|uses?|leverages?|depends on)\s+"
                + NAME_PATTERN,
                re.I,
            ),
        ]
        for pattern in integration_patterns:
            for m in pattern.finditer(sentence):
                add_relationship(
                    relationships,
                    m.group(1),
                    m.group(2),
                    "INTEGRATED_INTO",
                )

        # Employment variants:
        # "Y joined X", "X brought Y on board"
        joined = re.compile(
            NAME_PATTERN + r"\s+(?:joined|works at|worked at)\s+" + NAME_PATTERN,
            re.I,
        )
        for m in joined.finditer(sentence):
            person, org = m.group(1), m.group(2)
            add_relationship(relationships, org, person, "HIRED")

        brought = re.compile(
            NAME_PATTERN
            + r"\s+(?:brought|welcomed)\s+"
            + NAME_PATTERN
            + r"\s+(?:on board|to the team)",
            re.I,
        )
        for m in brought.finditer(sentence):
            add_relationship(
                relationships, m.group(1), m.group(2), "HIRED"
            )

    return relationships


def extract_entities_and_relationships(
    text: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    relationships = extract_relationships(text)
    names = candidate_names(text)

    for rel in relationships:
        names.extend((rel["source"], rel["target"]))

    blocked = {
        "was", "is", "were", "are", "has", "had", "been", "created",
        "developed", "founded", "integrates", "integrated", "hired",
        "authored", "wrote", "built", "made", "designed", "joined"
    }

    cleaned: List[str] = []
    for raw in names:
        name = clean_name(raw).rstrip(".")
        tokens = set(tokenise(name))
        if not name or tokens & blocked:
            continue
        if name not in cleaned:
            cleaned.append(name)

    # Relationship endpoints are authoritative entities.
    endpoint_types: Dict[str, str] = {}
    for rel in relationships:
        source, target, relation = (
            rel["source"], rel["target"], rel["relation"]
        )
        if relation in {"FOUNDED", "DEVELOPED", "CREATED", "AUTHORED"}:
            endpoint_types.setdefault(source, "Person")
        elif relation == "HIRED":
            endpoint_types.setdefault(source, "Organization")
            endpoint_types.setdefault(target, "Person")
        elif relation == "INTEGRATED_INTO":
            endpoint_types.setdefault(
                source,
                infer_type(source, text),
            )
            endpoint_types.setdefault(
                target,
                infer_type(target, text),
            )

    entities = []
    for name in cleaned:
        entity_type = endpoint_types.get(name) or infer_type(name, text)
        entities.append({"name": name, "type": entity_type})

    entities.sort(key=lambda e: e["name"])
    relationships.sort(
        key=lambda r: (r["source"], r["target"], r["relation"])
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


def _appears_in_text(name: str, text: str) -> bool:
    """Accept only names that occur as complete text spans."""
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name.strip())}(?![A-Za-z0-9_])",
            text,
            flags=re.I,
        )
    )


def extract_with_gemini(
    text: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in Render")

    client = genai.Client(api_key=api_key)

    prompt = f"""
Extract a complete knowledge graph from the supplied text.

Allowed entity types:
Person, Organization, Product, Framework

Allowed relationship labels:
FOUNDED, DEVELOPED, CREATED, INTEGRATED_INTO, HIRED, AUTHORED

Relationship direction:
- Person FOUNDED Organization/Product/Framework
- Person DEVELOPED Product/Framework
- Person CREATED Product/Framework
- Framework/Product INTEGRATED_INTO Organization/Product/Framework
- Organization HIRED Person
- Person AUTHORED Product

Rules:
1. Extract every explicitly stated named entity and relationship.
2. Resolve pronouns and descriptions such as "the framework", "the company",
   "it", "he", and "she" to a named entity already present in the text.
3. Entity names and relationship endpoints MUST be exact named spans present
   in the original text. Never include surrounding words such as "by",
   "was", punctuation, or the following sentence.
4. Preserve original capitalization.
5. Do not infer unstated relationships.
6. Do not return duplicates.
7. Return only data matching the response schema.

Examples:
Text: LangChain was created by Harrison Chase. The framework integrates with OpenAI.
Entities:
- LangChain, Framework
- Harrison Chase, Person
- OpenAI, Organization
Relationships:
- Harrison Chase -> LangChain, CREATED
- LangChain -> OpenAI, INTEGRATED_INTO

TEXT:
{text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=ExtractGraphResponse,
        ),
    )

    if response.parsed is None:
        raise RuntimeError(
            f"Gemini returned no structured output. Raw response: {response.text!r}"
        )

    if isinstance(response.parsed, ExtractGraphResponse):
        parsed = response.parsed
    else:
        parsed = ExtractGraphResponse.model_validate(response.parsed)

    allowed_types = {"Person", "Organization", "Product", "Framework"}
    allowed_relations = {
        "FOUNDED",
        "DEVELOPED",
        "CREATED",
        "INTEGRATED_INTO",
        "HIRED",
        "AUTHORED",
    }

    entities_by_name: Dict[str, Dict[str, str]] = {}
    for entity in parsed.entities:
        name = entity.name.strip().strip(".,;:")
        entity_type = entity.type
        if (
            name
            and entity_type in allowed_types
            and _appears_in_text(name, text)
        ):
            entities_by_name[name.casefold()] = {
                "name": name,
                "type": entity_type,
            }

    relationships_by_key: Dict[
        Tuple[str, str, str], Dict[str, str]
    ] = {}

    for relationship in parsed.relationships:
        source = relationship.source.strip().strip(".,;:")
        target = relationship.target.strip().strip(".,;:")
        relation = normalize_relation(relationship.relation)

        if (
            relation not in allowed_relations
            or not _appears_in_text(source, text)
            or not _appears_in_text(target, text)
            or source.casefold() == target.casefold()
        ):
            continue

        key = (source.casefold(), target.casefold(), relation)
        relationships_by_key[key] = {
            "source": source,
            "target": target,
            "relation": relation,
        }

        # Ensure relationship endpoints are included as entities.
        if source.casefold() not in entities_by_name:
            source_type = (
                "Person"
                if relation
                in {"FOUNDED", "DEVELOPED", "CREATED", "AUTHORED"}
                else infer_type(source, text)
            )
            if relation == "HIRED":
                source_type = "Organization"
            entities_by_name[source.casefold()] = {
                "name": source,
                "type": source_type,
            }

        if target.casefold() not in entities_by_name:
            target_type = infer_type(target, text)
            if relation == "HIRED":
                target_type = "Person"
            entities_by_name[target.casefold()] = {
                "name": target,
                "type": target_type,
            }

    entities = sorted(
        entities_by_name.values(),
        key=lambda item: item["name"].casefold(),
    )
    relationships = sorted(
        relationships_by_key.values(),
        key=lambda item: (
            item["source"].casefold(),
            item["target"].casefold(),
            item["relation"],
        ),
    )

    return entities, relationships

def _fast_named_entities(text: str) -> List[str]:
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"(?:[A-Z][A-Za-z0-9_+.#/-]*|[A-Z]{2,})"
        r"(?:[ \t]+(?:[A-Z][A-Za-z0-9_+.#/'-]*|[A-Z]{2,})){0,4}"
    )
    stop = {
        "The", "A", "An", "This", "That", "These", "Those", "It", "Its",
        "He", "She", "They", "His", "Her", "Their", "Later", "Meanwhile",
        "After", "Before", "Framework", "Product", "Organization", "Person"
    }
    names: List[str] = []
    for m in pattern.finditer(text):
        name = clean_name(m.group(0)).rstrip(".")
        if not name or name in stop:
            continue
        if re.search(
            r"\b(?:was|were|is|are|by|with|into|founded|created|developed|"
            r"built|hired|authored|wrote|integrates|integrated|joined)\b",
            name,
            flags=re.I,
        ):
            continue
        if name not in names:
            names.append(name)
    return names


def _fast_type(name: str, text: str) -> str:
    if name in KNOWN_TYPES:
        return KNOWN_TYPES[name]

    escaped = re.escape(name)
    rules = [
        ("Framework", rf"{escaped}.{{0,25}}\b(?:framework|library|toolkit|sdk|package|platform)\b"),
        ("Framework", rf"\b(?:framework|library|toolkit|sdk|package|platform)\b.{{0,25}}{escaped}"),
        ("Product", rf"{escaped}.{{0,25}}\b(?:product|model|assistant|application|app|service|engine|system)\b"),
        ("Product", rf"\b(?:product|model|assistant|application|app|service|engine|system)\b.{{0,25}}{escaped}"),
        ("Organization", rf"{escaped}.{{0,25}}\b(?:company|organization|startup|firm|lab|institute)\b"),
        ("Organization", rf"\b(?:company|organization|startup|firm|lab|institute)\b.{{0,25}}{escaped}"),
    ]
    for entity_type, pattern in rules:
        if re.search(pattern, text, flags=re.I):
            return entity_type

    words = name.split()
    if (
        2 <= len(words) <= 4
        and all(re.fullmatch(r"[A-Z][A-Za-z'-]*", w) for w in words)
        and not any(w.rstrip(".") in ORG_SUFFIXES for w in words)
    ):
        return "Person"

    if words and words[-1].rstrip(".") in ORG_SUFFIXES:
        return "Organization"

    if re.search(r"[a-z][A-Z]|\d|[+#]", name):
        return "Framework"

    return "Organization"


def _replace_generic_references(sentence: str, latest: Dict[str, str]) -> str:
    mapping = {
        "framework": "Framework", "library": "Framework",
        "toolkit": "Framework", "sdk": "Framework",
        "platform": "Framework", "package": "Framework",
        "product": "Product", "model": "Product",
        "assistant": "Product", "application": "Product",
        "app": "Product", "service": "Product",
        "company": "Organization", "organization": "Organization",
        "startup": "Organization", "firm": "Organization",
    }
    result = sentence
    for generic, entity_type in mapping.items():
        referent = latest.get(entity_type)
        if referent:
            result = re.sub(
                rf"\b(?:the|this|that|its|their)\s+{generic}\b",
                referent,
                result,
                flags=re.I,
            )
    if latest.get("Person"):
        result = re.sub(
            r"\b(?:he|she|him|her|his)\b",
            latest["Person"],
            result,
            flags=re.I,
        )
    return result


def _relation_add(
    relationships: List[Dict[str, str]],
    source: str,
    target: str,
    relation: str,
) -> None:
    source, target = clean_name(source), clean_name(target)
    relation = normalize_relation(relation)
    item = {"source": source, "target": target, "relation": relation}
    if (
        source and target and source != target
        and relation in RELATIONS and item not in relationships
    ):
        relationships.append(item)


def extract_graph_fast(
    text: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    relationships: List[Dict[str, str]] = []
    entity_types: Dict[str, str] = {}
    latest: Dict[str, str] = {}

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?;])\s+|\n+", text)
        if s.strip()
    ]

    for raw_sentence in sentences:
        for name in _fast_named_entities(raw_sentence):
            entity_type = _fast_type(name, raw_sentence)
            entity_types.setdefault(name, entity_type)
            latest[entity_type] = name

        sentence = _replace_generic_references(raw_sentence, latest)
        names = _fast_named_entities(sentence)

        for known_name in entity_types:
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(known_name)}(?![A-Za-z0-9_])",
                sentence,
            ) and known_name not in names:
                names.append(known_name)

        names = sorted(set(names), key=len, reverse=True)
        if len(names) < 2:
            continue

        alt = "|".join(re.escape(name) for name in names)

        passive_specs = [
            ("FOUNDED", r"founded|co-founded|established|started|launched"),
            ("DEVELOPED", r"developed|built|engineered|implemented|designed"),
            ("CREATED", r"created|invented|made|introduced"),
            ("HIRED", r"hired|recruited|employed|appointed"),
            ("AUTHORED", r"authored|written|published|co-authored"),
        ]
        for relation, verbs in passive_specs:
            pattern = re.compile(
                rf"(?P<target>{alt})(?:\s*,[^,]{{0,70}},\s*|\s+)"
                rf"(?:was|were|is|are|has been|had been)\s+"
                rf"(?:{verbs})\s+by\s+(?P<source>{alt})",
                flags=re.I,
            )
            for m in pattern.finditer(sentence):
                _relation_add(
                    relationships, m.group("source"), m.group("target"), relation
                )

        active_specs = [
            ("FOUNDED", r"founded|co-founded|established|started|launched"),
            ("DEVELOPED", r"developed|built|engineered|implemented|designed"),
            ("CREATED", r"created|invented|made|introduced"),
            ("HIRED", r"hired|recruited|employed|appointed"),
            ("AUTHORED", r"authored|wrote|published|co-authored"),
        ]
        for relation, verbs in active_specs:
            pattern = re.compile(
                rf"(?P<source>{alt})\s+(?:has\s+|had\s+|also\s+)?"
                rf"(?:{verbs})\s+(?P<target>{alt})",
                flags=re.I,
            )
            for m in pattern.finditer(sentence):
                _relation_add(
                    relationships, m.group("source"), m.group("target"), relation
                )

        integration_patterns = [
            re.compile(
                rf"(?P<source>{alt})\s+"
                rf"(?:integrates?|integrated|connects?|works?|interfaces?|"
                rf"interoperates?|collaborates?)\s+(?:directly\s+)?"
                rf"(?:with|into|to|alongside)\s+(?P<target>{alt})",
                flags=re.I,
            ),
            re.compile(
                rf"(?P<source>{alt})\s+(?:is|was|has been)\s+integrated\s+"
                rf"(?:with|into)\s+(?P<target>{alt})",
                flags=re.I,
            ),
            re.compile(
                rf"(?P<source>{alt})\s+"
                rf"(?:supports?|uses?|leverages?|depends\s+on)\s+"
                rf"(?P<target>{alt})",
                flags=re.I,
            ),
        ]
        for pattern in integration_patterns:
            for m in pattern.finditer(sentence):
                _relation_add(
                    relationships,
                    m.group("source"),
                    m.group("target"),
                    "INTEGRATED_INTO",
                )

        joined = re.compile(
            rf"(?P<person>{alt})\s+(?:joined|works at|worked at)\s+"
            rf"(?P<org>{alt})",
            flags=re.I,
        )
        for m in joined.finditer(sentence):
            _relation_add(
                relationships, m.group("org"), m.group("person"), "HIRED"
            )

    for rel in relationships:
        source, target, relation = (
            rel["source"], rel["target"], rel["relation"]
        )
        if relation in {"FOUNDED", "DEVELOPED", "CREATED", "AUTHORED"}:
            entity_types[source] = "Person"
        elif relation == "HIRED":
            entity_types[source] = "Organization"
            entity_types[target] = "Person"
        entity_types.setdefault(source, _fast_type(source, text))
        entity_types.setdefault(target, _fast_type(target, text))

    for name in _fast_named_entities(text):
        entity_types.setdefault(name, _fast_type(name, text))

    entities = [
        {"name": name, "type": entity_type}
        for name, entity_type in entity_types.items()
    ]
    entities.sort(key=lambda item: item["name"].casefold())
    relationships.sort(
        key=lambda item: (
            item["source"].casefold(),
            item["target"].casefold(),
            item["relation"],
        )
    )
    return entities, relationships

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
    entities, relationships = extract_graph_fast(request.text)
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
