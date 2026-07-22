from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_extract():
    text = (
        "LangChain was created by Harrison Chase. "
        "LangChain integrates with OpenAI."
    )
    response = client.post("/extract-graph", json={"chunk_id": "C001", "text": text})
    assert response.status_code == 200
    data = response.json()
    assert any(e["name"] == "LangChain" for e in data["entities"])
    assert any(
        r["source"] == "Harrison Chase"
        and r["target"] == "LangChain"
        and r["relation"] == "CREATED"
        for r in data["relationships"]
    )

def test_query():
    graph = {
        "entities": [
            {"name": "OpenAI", "type": "Organization"},
            {"name": "LangChain", "type": "Framework"},
            {"name": "Harrison Chase", "type": "Person"},
        ],
        "relationships": [
            {"source": "LangChain", "target": "OpenAI", "relation": "INTEGRATED_INTO"},
            {"source": "Harrison Chase", "target": "LangChain", "relation": "CREATED"},
        ],
    }
    response = client.post(
        "/graph-query",
        json={
            "question": "Who created the framework that integrates with OpenAI?",
            "graph": graph,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Harrison Chase"
    assert data["reasoning_path"] == ["OpenAI", "LangChain", "Harrison Chase"]
    assert data["hops"] == 2

def test_summary():
    response = client.post(
        "/community-summary",
        json={
            "community_id": "COM_001",
            "entities": ["LangChain", "Harrison Chase", "OpenAI"],
            "relationships": [
                {"source": "Harrison Chase", "target": "LangChain", "relation": "CREATED"},
                {"source": "LangChain", "target": "OpenAI", "relation": "INTEGRATED_INTO"},
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["community_id"] == "COM_001"
