"""
Knowledge Graph RAG with Verifiable Citations
=============================================
Streamlit app — multi-hop graph traversal, entity extraction via Ollama,
storage in Neo4j, and fully cited answers.
"""

import streamlit as st
import json
import re
import os
import hashlib
from typing import List, Tuple
from dataclasses import dataclass, field
from ollama import Client as OllamaClient
from neo4j import GraphDatabase

# ── Config ─────────────────────────────────────────────────────────────────
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


# ── Knowledge Graph Manager ────────────────────────────────────────────────

class KnowledgeGraphManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def test_connection(self) -> bool:
        try:
            with self.driver.session() as s:
                s.run("RETURN 1")
            return True
        except Exception:
            return False

    def clear_graph(self):
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def add_entity(self, entity: Entity):
        with self.driver.session() as s:
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

    def add_relationship(self, rel: Relationship):
        with self.driver.session() as s:
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

    def get_stats(self) -> dict:
        with self.driver.session() as s:
            nodes = s.run("MATCH (n) RETURN count(n) as c").single()["c"]
            rels  = s.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
            docs  = s.run("MATCH (n) RETURN collect(DISTINCT n.source_doc) as d").single()["d"]
        return {"nodes": nodes, "relationships": rels, "documents": [d for d in docs if d]}

    def find_related_entities(self, entity_name: str, hops: int = 2) -> List[dict]:
        with self.driver.session() as s:
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

    def semantic_search(self, query: str) -> List[dict]:
        words = [w.strip().lower() for w in query.split() if len(w.strip()) > 3]
        with self.driver.session() as s:
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

    def get_all_entities(self) -> List[dict]:
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (e:Entity)
                RETURN e.name AS name, e.type AS type,
                       e.description AS description, e.source_doc AS source
                ORDER BY e.type, e.name
                """
            )
            return [dict(r) for r in result]

    def get_all_relationships(self) -> List[dict]:
        with self.driver.session() as s:
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
        # Strip accidental markdown fences
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
        st.warning(f"⚠️ Entity extraction error: {exc}")
        return [], []


# ── Multi-hop RAG with Citations ───────────────────────────────────────────

def generate_answer_with_citations(
    query: str,
    graph: KnowledgeGraphManager,
    hops: int = 2,
    model: str = OLLAMA_MODEL,
) -> AnswerWithCitations:
    trace: List[str] = []
    citations: List[Citation] = []

    # Step 1 — Semantic search
    trace.append(f"🔍 **Step 1** — Searching knowledge graph for: `{query}`")
    seeds = graph.semantic_search(query)

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
        related = graph.find_related_entities(seed["name"], hops=hops)
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


# ── Streamlit UI ───────────────────────────────────────────────────────────

def render_connection_status(graph: KnowledgeGraphManager | None, connected: bool):
    if connected and graph:
        stats = graph.get_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Entities",      stats["nodes"])
        col2.metric("Relationships", stats["relationships"])
        col3.metric("Documents",     len(stats["documents"]))
    else:
        st.error("❌ Cannot connect to Neo4j. Check your settings in the sidebar.")


def main():
    st.set_page_config(
        page_title="MultiHop RAG — Knowledge Graph",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    code, .stCode { font-family: 'JetBrains Mono', monospace !important; }

    .block-container { padding-top: 2rem; }

    .rag-title {
        font-size: 1.9rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6c63ff, #00d4aa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }
    .rag-subtitle {
        color: #7070a0;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background: #13131f;
        border: 1px solid rgba(108,99,255,0.2);
        border-left: 3px solid #6c63ff;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
    }
    .citation-ref  { color: #6c63ff; font-weight: 700; font-size: 1rem; }
    .citation-doc  { color: #00d4aa; font-size: 0.8rem; font-weight: 600; }
    .citation-text { color: #b8b8cc; margin-top: 0.4rem; line-height: 1.6; }
    .entity-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-PERSON       { background: rgba(0,212,170,0.15); color: #00d4aa; }
    .badge-ORGANIZATION { background: rgba(245,200,66,0.15); color: #f5c842; }
    .badge-TECHNOLOGY   { background: rgba(255,107,107,0.15); color: #ff6b6b; }
    .badge-CONCEPT      { background: rgba(167,139,250,0.15); color: #a78bfa; }
    .badge-EVENT        { background: rgba(251,146,60,0.15);  color: #fb923c; }
    .badge-LOCATION     { background: rgba(34,211,238,0.15);  color: #22d3ee; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        neo4j_uri  = st.text_input("Neo4j URI",      NEO4J_URI)
        neo4j_user = st.text_input("Neo4j User",     NEO4J_USER)
        neo4j_pw   = st.text_input("Neo4j Password", NEO4J_PASSWORD, type="password")
        model      = st.selectbox("Ollama Model", ["llama3.2", "llama3.1", "mistral", "phi3", "gemma2"])
        hops       = st.slider("Traversal Hops", 1, 4, 2,
                               help="How many hops to traverse from seed entities")

        st.divider()

        # Connection test
        graph, connected = None, False
        try:
            graph = KnowledgeGraphManager(neo4j_uri, neo4j_user, neo4j_pw)
            connected = graph.test_connection()
        except Exception:
            pass

        if connected:
            st.success("✅ Neo4j connected")
        else:
            st.error("❌ Neo4j unreachable")

        st.divider()
        if connected and graph:
            if st.button("🗑️ Clear Entire Graph", type="secondary", use_container_width=True):
                graph.clear_graph()
                st.session_state.pop("documents", None)
                st.success("Graph cleared!")
                st.rerun()

    # ── Header ────────────────────────────────────────────────────────
    st.markdown('<div class="rag-title">🔍 MultiHop RAG — Knowledge Graph</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-subtitle">Build a structured knowledge graph from documents → traverse entity relationships → get answers with verifiable citations.</div>', unsafe_allow_html=True)

    render_connection_status(graph, connected)
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📄 Add Documents", "❓ Ask & Cite", "🔬 Explore Graph"])

    # ── TAB 1: Add Documents ──────────────────────────────────────────
    with tab1:
        st.subheader("Step 1 — Build Knowledge Graph from Documents")
        st.caption("Paste any document. The LLM extracts entities and relationships and stores them in Neo4j with full source provenance.")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            use_sample = st.checkbox("Use a sample document", value=True)
            if use_sample:
                doc_choice = st.selectbox("Choose sample:", list(SAMPLE_DOCS.keys()))
                doc_text   = st.text_area("Document text:", SAMPLE_DOCS[doc_choice], height=260)
                doc_name   = doc_choice
            else:
                doc_name = st.text_input("Document name:", placeholder="My Research Paper")
                doc_text = st.text_area("Paste document text:", height=260,
                                        placeholder="Paste any text here…")

            if st.button("🔨 Extract & Add to Knowledge Graph",
                         disabled=not connected, type="primary",
                         use_container_width=True):
                if not doc_text.strip():
                    st.warning("Please enter some document text first.")
                elif not doc_name.strip():
                    st.warning("Please give the document a name.")
                else:
                    with st.spinner("Extracting entities and relationships with LLM…"):
                        entities, relationships = extract_entities_with_llm(
                            doc_text, doc_name, model
                        )

                    if not entities:
                        st.error("No entities extracted. Check that Ollama is running and the model is pulled.")
                    else:
                        with st.spinner("Storing in Neo4j…"):
                            for e in entities:
                                graph.add_entity(e)
                            for r in relationships:
                                graph.add_relationship(r)

                        docs = st.session_state.get("documents", [])
                        if doc_name not in docs:
                            docs.append(doc_name)
                        st.session_state["documents"] = docs

                        st.success(f"✅ Extracted **{len(entities)}** entities and **{len(relationships)}** relationships from *{doc_name}*.")
                        st.rerun()

        with col_right:
            st.markdown("**Extraction preview will appear after processing**")
            if connected and graph:
                stats = graph.get_stats()
                if stats["nodes"]:
                    st.markdown(f"**Graph currently contains:**")
                    st.markdown(f"- `{stats['nodes']}` entities")
                    st.markdown(f"- `{stats['relationships']}` relationships")
                    if stats["documents"]:
                        st.markdown("**Indexed documents:**")
                        for d in stats["documents"]:
                            st.markdown(f"  📄 {d}")

        # Show extracted data after rerun
        if connected and graph and st.session_state.get("last_extracted"):
            entities, relationships = st.session_state["last_extracted"]
            with st.expander(f"📋 Last extracted: {len(entities)} entities, {len(relationships)} relationships"):
                for e in entities:
                    badge_cls = f"badge-{e.entity_type}"
                    st.markdown(
                        f'<span class="entity-badge {badge_cls}">{e.entity_type}</span> '
                        f'**{e.name}** — {e.description}',
                        unsafe_allow_html=True,
                    )
                st.divider()
                for r in relationships:
                    st.markdown(f"→ `{r.source}` **{r.relation_type}** `{r.target}`")

    # ── TAB 2: Ask & Cite ─────────────────────────────────────────────
    with tab2:
        st.subheader("Step 2 — Ask Questions with Verifiable Citations")

        docs_loaded = st.session_state.get("documents", [])
        if not docs_loaded and connected and graph:
            stats = graph.get_stats()
            docs_loaded = stats.get("documents", [])

        if not docs_loaded:
            st.info("💡 Add at least one document in the **Add Documents** tab first.")

        query = st.text_input(
            "Your question:",
            placeholder="Who developed GraphRAG and what organization are they from?",
        )

        example_queries = [
            "Who developed GraphRAG and what organization are they from?",
            "What companies use Neo4j?",
            "How is Acme Corp connected to Microsoft?",
            "Who founded Neo4j and when?",
        ]
        with st.expander("💡 Example questions"):
            for q in example_queries:
                if st.button(q, key=f"ex_{q}"):
                    st.session_state["prefill_query"] = q
                    st.rerun()

        if "prefill_query" in st.session_state:
            query = st.session_state.pop("prefill_query")

        ask_btn = st.button(
            "🔍 Ask with Citations",
            disabled=not connected or not query.strip(),
            type="primary",
        )

        if ask_btn and query.strip():
            with st.spinner("Traversing knowledge graph and generating answer…"):
                result = generate_answer_with_citations(query, graph, hops=hops, model=model)

            # Reasoning trace
            with st.expander("🧠 Reasoning Trace", expanded=False):
                for step in result.reasoning_trace:
                    st.markdown(step)

            # Answer
            st.markdown("### 💬 Answer")
            st.markdown(result.answer)

            # Citations
            st.markdown("### 📚 Source Citations")
            if result.citations:
                for cit in result.citations:
                    badge_map = {
                        "PERSON":       "badge-PERSON",
                        "ORGANIZATION": "badge-ORGANIZATION",
                        "TECHNOLOGY":   "badge-TECHNOLOGY",
                        "CONCEPT":      "badge-CONCEPT",
                        "EVENT":        "badge-EVENT",
                        "LOCATION":     "badge-LOCATION",
                    }
                    entity_type = ""
                    if connected and graph:
                        for ent in graph.get_all_entities():
                            if ent["name"] == cit.entity:
                                entity_type = ent.get("type", "")
                                break
                    badge_cls = badge_map.get(entity_type, "badge-CONCEPT")

                    st.markdown(
                        f"""<div class="citation-card">
                            <span class="citation-ref">{cit.ref}</span>
                            <span class="entity-badge {badge_cls}">{entity_type or 'ENTITY'}</span>
                            <strong>{cit.entity}</strong><br>
                            <span class="citation-doc">📄 {cit.source_document}</span>
                            <div class="citation-text">{cit.source_text[:400]}{'…' if len(cit.source_text) > 400 else ''}</div>
                            <div style="margin-top:0.4rem;color:#404060;font-size:0.75rem;">
                                Confidence: {cit.confidence:.0%}
                                {(' · Path: ' + ' → '.join(cit.reasoning_path[:3])) if cit.reasoning_path else ''}
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No specific citations could be extracted for this answer. "
                        "Try adding more documents to enrich the graph.")

    # ── TAB 3: Explore Graph ──────────────────────────────────────────
    with tab3:
        st.subheader("Step 3 — Explore the Knowledge Graph")

        if not connected:
            st.error("Connect to Neo4j to explore the graph.")
        else:
            stats = graph.get_stats()
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Entities",      stats["nodes"])
            m2.metric("Total Relationships", stats["relationships"])
            m3.metric("Indexed Documents",   len(stats["documents"]))

            st.divider()
            col_e, col_r = st.columns(2)

            with col_e:
                st.markdown("#### Entities by Type")
                entities = graph.get_all_entities()
                if entities:
                    # Group by type
                    by_type: dict = {}
                    for e in entities:
                        t = e.get("type", "UNKNOWN")
                        by_type.setdefault(t, []).append(e)

                    type_colors = {
                        "PERSON":       "#00d4aa",
                        "ORGANIZATION": "#f5c842",
                        "TECHNOLOGY":   "#ff6b6b",
                        "CONCEPT":      "#a78bfa",
                        "EVENT":        "#fb923c",
                        "LOCATION":     "#22d3ee",
                    }
                    for etype, ents in sorted(by_type.items()):
                        color = type_colors.get(etype, "#7070a0")
                        st.markdown(
                            f'<span style="color:{color};font-weight:600">'
                            f'● {etype} ({len(ents)})</span>',
                            unsafe_allow_html=True,
                        )
                        for ent in ents:
                            with st.expander(ent["name"], expanded=False):
                                st.caption(f"Source: {ent['source']}")
                                st.write(ent["description"])
                else:
                    st.info("No entities yet. Add documents first.")

            with col_r:
                st.markdown("#### Relationships")
                rels = graph.get_all_relationships()
                if rels:
                    for r in rels:
                        st.markdown(
                            f"`{r['source']}` "
                            f"**→ {r['rel_type']} →** "
                            f"`{r['target']}`"
                        )
                        if r.get("description"):
                            st.caption(r["description"])
                else:
                    st.info("No relationships yet.")

            st.divider()
            st.markdown("#### Open Neo4j Browser")
            neo4j_browser = neo4j_uri.replace("bolt://", "http://").replace("7687", "7474")
            st.markdown(f"🔗 [Open Neo4j Browser]({neo4j_browser}) — visualize the full graph interactively.")
            st.code(
                "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50",
                language="cypher",
            )


if __name__ == "__main__":
    main()
