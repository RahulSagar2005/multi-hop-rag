# MultiHop RAG — Knowledge Graph

A Flask-based Knowledge Graph RAG (Retrieval-Augmented Generation) application that combines graph-based storage with LLM-powered entity extraction and multi-hop reasoning to provide answers with verifiable citations.

## Features

- **Document Processing**: Paste any text and automatically extract entities (people, organizations, technologies, concepts) and relationships using an LLM
- **Multi-Hop Traversal**: Query the knowledge graph with configurable hop counts to find connected information
- **Verifiable Citations**: Get answers with inline citations ([1], [2], etc.) that link back to source documents
- **Graph Exploration**: Browse entities by type and explore relationships in an interactive interface
- **User Authentication**: Built-in user registration and login system

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Flask     │────▶│    Ollama    │────▶│    Neo4j    │
│   App       │◀────│   (LLLM)     │◀────│  (Graph DB) │
└─────────────┘     └──────────────┘     └─────────────┘
     │
     ▼
┌─────────────┐
│   SQLite    │
│  (Users)    │
└─────────────┘
```

## Prerequisites

1. **Python 3.10+**
2. **Neo4j Database** (local or cloud)
3. **Ollama** with a local LLM model

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Neo4j

**Option A: Docker (Recommended)**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

**Option B: Neo4j Desktop**
- Download from https://neo4j.com/download/
- Create a new database instance
- Note the bolt URI (default: `bolt://localhost:7687`)

### 3. Set Up Ollama

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
```

### 4. Configure Environment Variables (Optional)

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2
export SECRET_KEY=your-secret-key-change-in-prod
```

## Running the Application

```bash
python app.py
```

The application will start at `http://localhost:5001`

## Usage

### 1. Register/Login

- Navigate to the landing page
- Click "Get started" to register or "Sign in" if you have an account

### 2. Add Documents

- Go to the **Knowledge Graph** page from the dashboard
- In the "Add Documents" tab:
  - Choose a sample document or paste your own text
  - Click "Extract & Add to Knowledge Graph"
  - Wait for the LLM to extract entities and relationships

### 3. Ask Questions

- Switch to the "Ask & Cite" tab
- Enter your question (e.g., "Who developed GraphRAG?")
- Adjust the traversal hops (1-4) for deeper graph exploration
- Click "Ask with Citations"
- View the answer with inline citations and reasoning trace

### 4. Explore the Graph

- Go to the "Explore Graph" tab
- Browse entities grouped by type (PERSON, ORGANIZATION, TECHNOLOGY, etc.)
- View all relationships between entities
- Clear the graph if needed

## API Endpoints

All API endpoints require authentication.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/extract-entities` | POST | Extract entities from document text |
| `/api/ask` | POST | Ask a question and get cited answer |
| `/api/graph/stats` | GET | Get graph statistics |
| `/api/graph/entities` | GET | Get all entities |
| `/api/graph/relationships` | GET | Get all relationships |
| `/api/graph/clear` | POST | Clear the entire graph |

### Example API Usage

```bash
# Extract entities
curl -X POST http://localhost:5001/api/extract-entities \
  -H "Content-Type: application/json" \
  -d '{"doc_name": "Test", "text": "John works at Google."}'

# Ask a question
curl -X POST http://localhost:5001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Who works at Google?", "hops": 2}'
```

## Sample Documents

The app includes three sample documents:

1. **AI Research Paper** — About GraphRAG and Microsoft Research
2. **Company Report** — About Acme Corp and its founders
3. **Neo4j Documentation** — About Neo4j features and history

## Project Structure

```
multihop_rag/
├── app.py                 # Main Flask application
├── knowledge_graph_rag.py # Original Streamlit app (reference)
├── tests.py               # Test suite
├── requirements.txt       # Python dependencies
├── database/
│   ├── __init__.py
│   └── db.py             # SQLite database helpers
├── template/
│   ├── base.html         # Base template
│   ├── landing.html      # Landing page
│   ├── login.html        # Login page
│   ├── register.html     # Registration page
│   ├── dashboard.html    # User dashboard
│   ├── profile.html      # User profile
│   └── knowledge_graph.html # Knowledge Graph UI
├── static/
│   ├── css/
│   │   └── style.css     # Stylesheet
│   └── js/
│       └── main.js       # JavaScript utilities
└── docker-compose.yml    # Docker configuration
```

## Docker Support

Run the entire stack with Docker Compose:

```bash
docker-compose up -d
```

This starts:
- Neo4j database
- Flask application

## Troubleshooting

### Neo4j Connection Failed
- Ensure Neo4j is running: `docker ps | grep neo4j`
- Check credentials in environment variables
- Verify the bolt port (7687) is accessible

### Ollama Errors
- Ensure Ollama is running: `ollama list`
- Pull the model: `ollama pull llama3.2`
- Check Ollama host configuration

### Entity Extraction Returns Empty
- Verify Ollama is responding: `ollama run llama3.2 "hello"`
- Try a different model in the dropdown
- Check that the document text is not empty

## License

MIT
