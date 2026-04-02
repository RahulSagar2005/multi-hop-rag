"""
Flask application for MultiHop RAG with Knowledge Graph
Combines user authentication with graph-based RAG functionality
"""
from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from database.db import init_db, create_user, get_user_by_email, get_user_by_id
import os
import hashlib
import re
import json
from typing import List, Tuple
from dataclasses import dataclass, field, asdict
from ollama import Client as OllamaClient
from neo4j import GraphDatabase

app = Flask(__name__, template_folder='template')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-in-prod')

with app.app_context():
    init_db()  # ← add this

# ── Configuration ────────────────────────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST",    "http://localhost:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",   "llama3.2")

ollama_client = OllamaClient(host=OLLAMA_HOST)

SAMPLE_DOCS = {
    "AI Research Paper": """
        GraphRAG is a technique developed by Microsoft Research that combines knowledge graphs
        with retrieval-augmented generation. Unlike traditional RAG which uses vector similarity,
        GraphRAG builds a structured knowledge graph from documents, enabling multi-hop reasoning.
        The technique was introduced by researchers including Darren Edge and Ha Trinh.
        GraphRAG excels at answering complex questions that require connecting information
        from multiple sources, such as understanding relationships between research projects.
        It was published as a paper titled "From Local to Global: A Graph RAG Approach to
        Query-Focused Summarization" and has been widely adopted in enterprise AI systems.
    """,
    "Company Report": """
        Acme Corp was founded in 2020 by Jane Smith and John Doe in San Francisco.
        The company develops AI-powered analytics tools for enterprise customers.
        Their flagship product, DataSense, uses machine learning to analyze business data.
        Jane Smith previously worked at Google as a senior engineer on the TensorFlow team.
        John Doe was a co-founder of StartupX, which was acquired by Microsoft in 2019.
        Acme Corp raised $50 million in Series B funding led by Sequoia Capital in 2023.
        The company partners with Neo4j to power their graph-based analytics features.
    """,
    "Neo4j Documentation": """
        Neo4j is a native graph database built to leverage data relationships.
        It uses the Cypher query language, designed specifically for graph traversal.
        Neo4j supports ACID transactions and is used for fraud detection, recommendation
        engines, knowledge graphs, and network analysis. The APOC plugin library extends
        Neo4j with hundreds of additional procedures and functions. Neo4j Aura is the
        fully managed cloud version, while the community edition is open source.
        Emil Eifrem co-founded Neo4j in 2007 and serves as CEO.
    """,
}


# ── Data Models ────────────────────────────────────────────────────────────

@dataclass
class Entity:
    id: str
    name: str
    entity_type: str
    description: str
    source_doc: str
    source_chunk: str


@dataclass
class Relationship:
    source: str
    target: str
    relation_type: str
    description: str
    source_doc: str


@dataclass
class Citation:
    ref: str
    source_document: str
    source_text: str
    entity: str
    confidence: float
    reasoning_path: List[str] = field(default_factory=list)


@dataclass
class AnswerWithCitations:
    answer: str
    citations: List[Citation]
    reasoning_trace: List[str]


# ── Database Initialization ───────────────────────────────────────────────

def get_graph_driver(uri=None, user=None, password=None):
    """Get Neo4j driver with optional override params."""
    uri = uri or NEO4J_URI
    user = user or NEO4J_USER
    password = password or NEO4J_PASSWORD
    return GraphDatabase.driver(uri, auth=(user, password))


# ── Helpers ───────────────────────────────────────────────────────────────

def current_user():
    uid = session.get('user_id')
    if uid:
        return get_user_by_id(uid)
    return None


def get_graph_stats(driver):
    """Get knowledge graph statistics."""
    with driver.session() as s:
        nodes = s.run("MATCH (n) RETURN count(n) as c").single()["c"]
        rels = s.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        docs = s.run("MATCH (n) RETURN collect(DISTINCT n.source_doc) as d").single()["d"]
    return {"nodes": nodes, "relationships": rels, "documents": [d for d in docs if d]}


# ── Knowledge Graph Operations ────────────────────────────────────────────

def clear_graph(driver):
    """Delete all nodes and relationships."""
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")


def add_entity(driver, entity: Entity):
    """Add an entity to the graph."""
    with driver.session() as s:
        s.run(
            """
            MERGE (e:Entity {id: $id})
            SET e.name        = $name,
                e.type        = $entity_type,
                e.description = $description,
                e.source_doc  = $source_doc,
                e.source_chunk= $source_chunk
            """,
            id=entity.id, name=entity.name, entity_type=entity.entity_type,
            description=entity.description, source_doc=entity.source_doc,
            source_chunk=entity.source_chunk,
        )


def add_relationship(driver, rel: Relationship):
    """Add a relationship to the graph."""
    with driver.session() as s:
        s.run(
            """
            MATCH (a:Entity {name: $source})
            MATCH (b:Entity {name: $target})
            MERGE (a)-[r:RELATES_TO {type: $rel_type}]->(b)
            SET r.description = $description,
                r.source_doc  = $source_doc
            """,
            source=rel.source, target=rel.target, rel_type=rel.relation_type,
            description=rel.description, source_doc=rel.source_doc,
        )


def find_related_entities(driver, entity_name: str, hops: int = 2) -> List[dict]:
    """Find entities related to a given entity via multi-hop traversal."""
    with driver.session() as s:
        result = s.run(
            f"""
            MATCH path = (start:Entity)-[*1..{hops}]-(related:Entity)
            WHERE toLower(start.name) CONTAINS toLower($name)
               OR toLower(start.description) CONTAINS toLower($name)
            RETURN DISTINCT
                related.name         AS name,
                related.description  AS description,
                related.source_doc   AS source,
                related.source_chunk AS chunk,
                related.type         AS type,
                [r IN relationships(path) | r.description] AS path_descriptions
            LIMIT 25
            """,
            name=entity_name,
        )
        return [dict(r) for r in result]


def semantic_search(driver, query: str) -> List[dict]:
    """Search for entities matching query keywords."""
    words = [w.strip().lower() for w in query.split() if len(w.strip()) > 3]
    with driver.session() as s:
        result = s.run(
            """
            MATCH (e:Entity)
            WHERE ANY(word IN $words WHERE
                toLower(e.name)        CONTAINS word OR
                toLower(e.description) CONTAINS word
            )
            RETURN e.name         AS name,
                   e.description  AS description,
                   e.source_doc   AS source,
                   e.source_chunk AS chunk,
                   e.type         AS type
            LIMIT 10
            """,
            words=words,
        )
        return [dict(r) for r in result]


def get_all_entities(driver) -> List[dict]:
    """Get all entities from the graph."""
    with driver.session() as s:
        result = s.run(
            """
            MATCH (e:Entity)
            RETURN e.name AS name, e.type AS type,
                   e.description AS description, e.source_doc AS source
            ORDER BY e.type, e.name
            """
        )
        return [dict(r) for r in result]


def get_all_relationships(driver) -> List[dict]:
    """Get all relationships from the graph."""
    with driver.session() as s:
        result = s.run(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            RETURN a.name AS source, r.type AS rel_type,
                   b.name AS target, r.description AS description
            """
        )
        return [dict(r) for r in result]


# ── LLM Entity Extraction ──────────────────────────────────────────────────

def extract_entities_with_llm(
    text: str, source_doc: str, model: str = OLLAMA_MODEL
) -> Tuple[List[Entity], List[Relationship]]:
    """Extract entities and relationships from text using LLM."""
    prompt = f"""Analyze the following text carefully and extract structured knowledge.

Extract:
1. KEY ENTITIES — people, organizations, technologies, concepts, events, locations.
2. RELATIONSHIPS — directed connections between those entities.

For each entity provide:
  - name: canonical name (e.g. "Microsoft Research", not "it" or "they")
  - type: one of PERSON | ORGANIZATION | TECHNOLOGY | CONCEPT | EVENT | LOCATION
  - description: one concise sentence from the text describing this entity

For each relationship provide:
  - source: exact name of source entity
  - target: exact name of target entity
  - type: short snake_case verb phrase (e.g. WORKS_FOR, CREATED, USES, PART_OF, LOCATED_IN, FUNDED_BY)
  - description: one sentence explaining the relationship

TEXT:
\"\"\"
{text.strip()}
\"\"\"

Respond ONLY with valid JSON — no markdown, no explanation:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "description": "..."}}
  ]
}}"""

    try:
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        raw = response["message"]["content"]
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)

        entities = []
        for e in data.get("entities", []):
            if not e.get("name"):
                continue
            eid = hashlib.md5(f"{e['name']}_{source_doc}".encode()).hexdigest()[:12]
            entities.append(Entity(
                id=eid,
                name=e["name"],
                entity_type=e.get("type", "CONCEPT"),
                description=e.get("description", ""),
                source_doc=source_doc,
                source_chunk=text.strip()[:300] + ("..." if len(text) > 300 else ""),
            ))

        relationships = []
        entity_names = {en.name for en in entities}
        for r in data.get("relationships", []):
            if r.get("source") in entity_names and r.get("target") in entity_names:
                relationships.append(Relationship(
                    source=r["source"],
                    target=r["target"],
                    relation_type=r.get("type", "RELATED_TO"),
                    description=r.get("description", ""),
                    source_doc=source_doc,
                ))

        return entities, relationships

    except Exception as exc:
        return [], []


# ── Multi-hop RAG with Citations ───────────────────────────────────────────

def generate_answer_with_citations(
    query: str,
    driver,
    hops: int = 2,
    model: str = OLLAMA_MODEL,
) -> AnswerWithCitations:
    """Generate an answer with citations by traversing the knowledge graph."""
    trace: List[str] = []
    citations: List[Citation] = []

    # Step 1 — Semantic search
    trace.append(f"🔍 **Step 1** — Searching knowledge graph for: `{query}`")
    seeds = semantic_search(driver, query)

    if not seeds:
        trace.append("❌ No matching entities found in the graph.")
        return AnswerWithCitations(
            answer="I couldn't find relevant information in the knowledge graph. "
                   "Please add some documents first.",
            citations=[],
            reasoning_trace=trace,
        )

    trace.append(f"📍 Found **{len(seeds)}** seed entities: "
                 + ", ".join(f"`{s['name']}`" for s in seeds[:5]))

    # Step 2 — Multi-hop expansion
    trace.append(f"🔗 **Step 2** — Expanding {hops} hops from each seed entity…")
    all_context: List[dict] = []
    seen_names: set = set()

    for seed in seeds[:4]:
        related = find_related_entities(driver, seed["name"], hops=hops)
        new = [r for r in related if r["name"] not in seen_names]
        for r in new:
            seen_names.add(r["name"])
            all_context.append(r)
        trace.append(f"   └─ `{seed['name']}` → {len(new)} new related entities")

    # Also include seeds themselves
    for s in seeds:
        if s["name"] not in seen_names:
            all_context.append(s)
            seen_names.add(s["name"])

    trace.append(f"📊 **Step 3** — Built context from **{len(all_context)}** entities")

    # Step 3 — Build numbered source map
    context_lines: List[str] = []
    source_map: dict = {}

    for i, ctx in enumerate(all_context):
        ref = f"[{i + 1}]"
        context_lines.append(
            f"{ref} Entity: {ctx['name']} ({ctx.get('type','?')})\n"
            f"    Description: {ctx['description']}\n"
            f"    Source: {ctx['source']}"
        )
        source_map[ref] = ctx

    context_text = "\n\n".join(context_lines)

    # Step 4 — Generate answer
    trace.append("🤖 **Step 4** — Generating cited answer with LLM…")

    answer_prompt = f"""You are a research assistant. Answer the question using ONLY the provided knowledge graph context.

RULES:
- Cite every factual claim with its reference number [N].
- Use multiple citations per sentence when the claim draws from multiple sources.
- Do not invent information not present in the context.
- Be concise and precise.

KNOWLEDGE GRAPH CONTEXT:
{context_text}

QUESTION: {query}

Write a thorough answer with inline citations [1], [2], etc.:"""

    try:
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": answer_prompt}],
        )
        answer = response["message"]["content"].strip()

        # Step 5 — Extract and build citation objects
        refs_used = sorted(set(re.findall(r"\[(\d+)\]", answer)), key=int)
        for ref_num in refs_used:
            ref_key = f"[{ref_num}]"
            if ref_key in source_map:
                ctx = source_map[ref_key]
                citations.append(Citation(
                    ref=ref_key,
                    source_document=ctx.get("source", "Unknown"),
                    source_text=ctx.get("chunk", ctx.get("description", "")),
                    entity=ctx.get("name", ""),
                    confidence=0.90,
                    reasoning_path=ctx.get("path_descriptions") or [],
                ))

        trace.append(f"✅ Answer generated with **{len(citations)}** verified citations.")

        return AnswerWithCitations(answer=answer, citations=citations, reasoning_trace=trace)

    except Exception as exc:
        trace.append(f"❌ LLM error: {exc}")
        return AnswerWithCitations(
            answer=f"Error generating answer: {exc}",
            citations=[],
            reasoning_trace=trace,
        )


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def landing():
    return render_template("landing.html", user=current_user())


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user():
        return redirect(url_for('dashboard'))

    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            error = "All fields are required."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        elif get_user_by_email(email):
            error = "An account with that email already exists."
        else:
            hashed = generate_password_hash(password)
            create_user(name, email, hashed)
            user = get_user_by_email(email)
            session['user_id'] = user['id']
            flash("Welcome to MultiHop RAG!", "success")
            return redirect(url_for('dashboard'))

    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for('dashboard'))

    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_user_by_email(email)

        if not user or not check_password_hash(user['password'], password):
            error = "Invalid email or password."
        else:
            session['user_id'] = user['id']
            flash(f"Welcome back, {user['name'].split()[0]}!", "success")
            return redirect(url_for('dashboard'))

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    flash("You've been signed out.", "info")
    return redirect(url_for('landing'))


@app.route("/dashboard")
def dashboard():
    user = current_user()
    if not user:
        return redirect(url_for('login'))

    # Test Neo4j connection and get stats
    neo4j_connected = False
    graph_stats = None
    try:
        driver = get_graph_driver()
        with driver.session() as s:
            s.run("RETURN 1")
        neo4j_connected = True
        graph_stats = get_graph_stats(driver)
        driver.close()
    except Exception:
        neo4j_connected = False

    return render_template("dashboard.html", user=user, neo4j_connected=neo4j_connected, graph_stats=graph_stats)


@app.route("/profile")
def profile():
    user = current_user()
    if not user:
        return redirect(url_for('login'))
    return render_template("profile.html", user=user)


# ── Knowledge Graph Routes ────────────────────────────────────────────────

@app.route("/knowledge-graph")
def knowledge_graph():
    """Main knowledge graph interface."""
    user = current_user()
    if not user:
        return redirect(url_for('login'))

    # Test Neo4j connection
    neo4j_connected = False
    graph_stats = None
    try:
        driver = get_graph_driver()
        with driver.session() as s:
            s.run("RETURN 1")
        neo4j_connected = True
        graph_stats = get_graph_stats(driver)
        driver.close()
    except Exception:
        neo4j_connected = False

    return render_template("knowledge_graph.html", user=user,
                         neo4j_connected=neo4j_connected,
                         graph_stats=graph_stats,
                         sample_docs=list(SAMPLE_DOCS.keys()))


@app.route("/api/extract-entities", methods=["POST"])
def api_extract_entities():
    """Extract entities and relationships from document text."""
    user = current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    doc_name = data.get("doc_name", "Untitled")
    doc_text = data.get("text", "")
    model = data.get("model", OLLAMA_MODEL)

    if not doc_text.strip():
        return jsonify({"error": "No text provided"}), 400

    entities, relationships = extract_entities_with_llm(doc_text, doc_name, model)

    # Store in database
    try:
        driver = get_graph_driver()
        for e in entities:
            add_entity(driver, e)
        for r in relationships:
            add_relationship(driver, r)
        driver.close()
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    return jsonify({
        "entities": [asdict(e) for e in entities],
        "relationships": [asdict(r) for r in relationships]
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Ask a question and get an answer with citations."""
    user = current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    query = data.get("query", "")
    hops = data.get("hops", 2)
    model = data.get("model", OLLAMA_MODEL)

    if not query.strip():
        return jsonify({"error": "No query provided"}), 400

    try:
        driver = get_graph_driver()
        result = generate_answer_with_citations(query, driver, hops=hops, model=model)
        driver.close()

        return jsonify({
            "answer": result.answer,
            "citations": [asdict(c) for c in result.citations],
            "reasoning_trace": result.reasoning_trace
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/stats")
def api_graph_stats():
    """Get knowledge graph statistics."""
    user = current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        driver = get_graph_driver()
        stats = get_graph_stats(driver)
        driver.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/entities")
def api_graph_entities():
    """Get all entities from the graph."""
    user = current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        driver = get_graph_driver()
        entities = get_all_entities(driver)
        driver.close()
        return jsonify({"entities": entities})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/relationships")
def api_graph_relationships():
    """Get all relationships from the graph."""
    user = current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        driver = get_graph_driver()
        relationships = get_all_relationships(driver)
        driver.close()
        return jsonify({"relationships": relationships})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/clear", methods=["POST"])
def api_graph_clear():
    """Clear all data from the graph."""
    user = current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        driver = get_graph_driver()
        clear_graph(driver)
        driver.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5001)
