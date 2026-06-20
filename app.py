import os
import sys
import time
import uuid
import math
import json
import sqlite3
import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException, Query

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_benchmark.db")

# ---------------------------------------------------------
# 1. Pydantic Schemas (Matching User Dataset Format)
# ---------------------------------------------------------
class RAGStoreItem(BaseModel):
    id: Any
    snapshot_id: Optional[int] = None
    user_id: Optional[int] = None
    text_chunk: str
    severity: Optional[str] = None
    created_at: Optional[str] = None
    embedding: Optional[List[float]] = None
    graph_nodes: Optional[List[str]] = None

class RAGStoreRequest(BaseModel):
    items: List[RAGStoreItem]

class RAGStoreResponse(BaseModel):
    message: str
    stored_linear: bool
    stored_vector: bool
    stored_graph: bool
    latency_ms: Dict[str, float]

class RAGRetrieveRequest(BaseModel):
    query: str
    query_vector: Optional[List[float]] = None

class RAGRetrieveResponse(BaseModel):
    linear_results: Optional[List[Dict[str, Any]]] = None
    vector_results: Optional[List[Dict[str, Any]]] = None
    graph_results: Optional[List[Dict[str, Any]]] = None
    latency_ms: Dict[str, float]

# ---------------------------------------------------------
# 2. SQLite Initialization & Seeding
# ---------------------------------------------------------
def initialize_db() -> None:
    """Creates the SQLite database schema and seeds initial clinical nodes/edges."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Drop existing tables if the schema is old (missing 'severity')
    try:
        cursor.execute("PRAGMA table_info(snapshots);")
        columns = [r[1] for r in cursor.fetchall()]
        if columns and "severity" not in columns:
            cursor.execute("DROP TABLE IF EXISTS snapshot_embeddings;")
            cursor.execute("DROP TABLE IF EXISTS patient_nodes;")
            cursor.execute("DROP TABLE IF EXISTS snapshots;")
    except Exception:
        pass
    
    # 1. Snapshots (Linear Store)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id TEXT PRIMARY KEY,
            snapshot_id INTEGER,
            user_id INTEGER NOT NULL,
            text_chunk TEXT NOT NULL,
            severity TEXT,
            created_at TIMESTAMP
        );
    """)
    
    # 2. Vector Store (Stored as JSON string representation)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshot_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            embedding TEXT NOT NULL,
            FOREIGN KEY (snapshot_id) REFERENCES snapshots (id) ON DELETE CASCADE
        );
    """)
    
    # 3. Clinical Nodes (Knowledge Graph nodes)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clinical_nodes (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL
        );
    """)
    
    # 4. Clinical Edges (Knowledge Graph edges)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clinical_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relationship TEXT NOT NULL,
            FOREIGN KEY (source) REFERENCES clinical_nodes (id) ON DELETE CASCADE,
            FOREIGN KEY (target) REFERENCES clinical_nodes (id) ON DELETE CASCADE
        );
    """)
    
    # 5. Patient Nodes mapping
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            node_id TEXT NOT NULL,
            relationship TEXT NOT NULL,
            FOREIGN KEY (node_id) REFERENCES clinical_nodes (id) ON DELETE CASCADE
        );
    """)
    
    conn.commit()
    
    # Seed standard clinical nodes and edges if empty or outdated
    cursor.execute("SELECT COUNT(*) FROM clinical_nodes;")
    if cursor.fetchone()[0] < 10:
        cursor.execute("DELETE FROM clinical_edges;")
        cursor.execute("DELETE FROM clinical_nodes;")
        
        nodes = [
            ("headache", "Symptom"),
            ("high bp", "Symptom"),
            ("tension-type headache", "Condition"),
            ("hypertension", "Condition"),
            ("rest", "Treatment"),
            ("monitor bp", "Treatment"),
            ("amlodipine", "Treatment"),
            ("anxiety", "Symptom"),
            ("nausea", "Symptom"),
            ("sweating", "Symptom"),
            ("shortness of breath", "Symptom"),
            ("chest pain", "Symptom"),
            ("heart attack", "Condition")
        ]
        
        edges = [
            ("tension-type headache", "headache", "causes"),
            ("hypertension", "high bp", "causes"),
            ("tension-type headache", "rest", "treatment"),
            ("hypertension", "monitor bp", "treatment"),
            ("hypertension", "amlodipine", "treatment"),
            ("heart attack", "chest pain", "causes"),
            ("heart attack", "nausea", "causes"),
            ("heart attack", "sweating", "causes"),
            ("heart attack", "shortness of breath", "causes")
        ]
        
        cursor.executemany("INSERT OR IGNORE INTO clinical_nodes (id, type) VALUES (?, ?);", nodes)
        cursor.executemany("INSERT OR IGNORE INTO clinical_edges (source, target, relationship) VALUES (?, ?, ?);", edges)
        conn.commit()
        
    conn.close()

# Helper: Cosine Similarity in pure python
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        return 0.0
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = math.sqrt(sum(x * x for x in v1))
    magnitude2 = math.sqrt(sum(x * x for x in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

# ---------------------------------------------------------
# 3. Store Implementation
# ---------------------------------------------------------
def store_rag_data_local(
    payload: List[RAGStoreItem],
    user_id: int,
    use_linear: bool,
    use_vector: bool,
    use_graph: bool
) -> RAGStoreResponse:
    initialize_db()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    latency = {"linear": 0.0, "vector": 0.0, "graph": 0.0, "total": 0.0}
    stored_linear = False
    stored_vector = False
    stored_graph = False
    
    t_start = time.perf_counter()
    
    for item in payload:
        snapshot_id = str(item.id)
        item_user_id = item.user_id if item.user_id is not None else user_id
        
        # 1. Base snapshot (Linear Store / Relational)
        t_linear_start = time.perf_counter()
        if use_linear or use_vector or use_graph:
            cursor.execute(
                """
                INSERT OR REPLACE INTO snapshots (id, snapshot_id, user_id, text_chunk, severity, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (snapshot_id, item.snapshot_id, item_user_id, item.text_chunk, item.severity, item.created_at)
            )
            stored_linear = use_linear
            latency["linear"] += (time.perf_counter() - t_linear_start) * 1000
            
        # 2. Vector Store
        if use_vector:
            t_vec_start = time.perf_counter()
            if item.embedding:
                # delete old embedding first to avoid duplicate keys if replaced
                cursor.execute("DELETE FROM snapshot_embeddings WHERE snapshot_id = ?;", (snapshot_id,))
                cursor.execute(
                    "INSERT INTO snapshot_embeddings (snapshot_id, embedding) VALUES (?, ?);",
                    (snapshot_id, json.dumps(item.embedding))
                )
                stored_vector = True
            latency["vector"] += (time.perf_counter() - t_vec_start) * 1000
            
        # 3. Graph Store
        if use_graph:
            t_graph_start = time.perf_counter()
            matched_nodes = []
            if item.graph_nodes:
                matched_nodes = item.graph_nodes
            else:
                text_lower = item.text_chunk.lower()
                cursor.execute("SELECT id FROM clinical_nodes;")
                all_node_ids = [r[0] for r in cursor.fetchall()]
                for nid in all_node_ids:
                    if nid.lower() in text_lower:
                        matched_nodes.append(nid)
                        
            for node in matched_nodes:
                cursor.execute("INSERT OR IGNORE INTO clinical_nodes (id, type) VALUES (?, 'Symptom');", (node,))
                # delete existing association to avoid duplication if re-stored
                cursor.execute("DELETE FROM patient_nodes WHERE user_id = ? AND node_id = ?;", (item_user_id, node))
                cursor.execute(
                    "INSERT INTO patient_nodes (user_id, node_id, relationship) VALUES (?, ?, 'suffers_from');",
                    (item_user_id, node)
                )
                stored_graph = True
            latency["graph"] += (time.perf_counter() - t_graph_start) * 1000
            
    conn.commit()
    conn.close()
    
    latency["linear"] = round(latency["linear"], 2)
    latency["vector"] = round(latency["vector"], 2)
    latency["graph"] = round(latency["graph"], 2)
    latency["total"] = round((time.perf_counter() - t_start) * 1000, 2)
    
    return RAGStoreResponse(
        message=f"Successfully stored {len(payload)} item(s) in local SQLite database.",
        stored_linear=stored_linear,
        stored_vector=stored_vector,
        stored_graph=stored_graph,
        latency_ms=latency
    )

# ---------------------------------------------------------
# 4. Retrieve Implementation
# ---------------------------------------------------------
def retrieve_rag_data_local(
    payload: RAGRetrieveRequest,
    user_id: int,
    use_linear: bool,
    use_vector: bool,
    use_graph: bool,
    similarity_threshold: float = 0.70
) -> RAGRetrieveResponse:
    initialize_db()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    latency = {"linear": 0.0, "vector": 0.0, "graph": 0.0, "total": 0.0}
    linear_results = None
    vector_results = None
    graph_results = None
    
    t_start = time.perf_counter()
    
    # 1. Linear Read
    if use_linear:
        t_lin_start = time.perf_counter()
        cursor.execute(
            "SELECT id, snapshot_id, text_chunk, severity, created_at FROM snapshots WHERE user_id = ? ORDER BY created_at DESC LIMIT 5;",
            (user_id,)
        )
        linear_results = []
        for r in cursor.fetchall():
            linear_results.append({
                "id": r[0],
                "snapshot_id": r[1],
                "text_chunk": r[2],
                "severity": r[3],
                "created_at": str(r[4])
            })
        latency["linear"] = round((time.perf_counter() - t_lin_start) * 1000, 2)
        
    # 2. Vector Read (In-memory Cosine Similarity with Similarity Thresholding & Time Decay)
    if use_vector:
        t_vec_start = time.perf_counter()
        if payload.query_vector:
            cursor.execute("""
                SELECT s.id, s.snapshot_id, s.text_chunk, s.severity, s.created_at, e.embedding
                FROM snapshot_embeddings e
                JOIN snapshots s ON e.snapshot_id = s.id
                WHERE s.user_id = ?;
            """, (user_id,))
            candidates = cursor.fetchall()
            
            scored_candidates = []
            for id_val, snap_id, text_chunk, severity, created_at, emb_str in candidates:
                emb = json.loads(emb_str)
                sim = cosine_similarity(payload.query_vector, emb)
                
                # Apply time-decay decay boost (recency weighting)
                decay_factor = 1.0
                if created_at:
                    try:
                        from datetime import datetime, timezone
                        dt_str = str(created_at).replace(" ", "T")
                        if "+" in dt_str:
                            dt = datetime.fromisoformat(dt_str)
                        else:
                            dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
                        
                        now = datetime.now(timezone.utc)
                        elapsed_days = (now - dt).days
                        # Half-life = 180 days (decays similarity score slightly over time)
                        # We apply a decay floor of 0.5 to prevent old matches from completely disappearing
                        decay_factor = math.exp(-max(0, elapsed_days) / 180)
                        decay_factor = max(0.5, decay_factor)
                    except Exception:
                        pass
                
                final_score = round(sim * decay_factor, 4)
                
                # Filter out candidates that do not meet the minimum similarity threshold
                if sim >= similarity_threshold:
                    scored_candidates.append({
                        "id": id_val,
                        "snapshot_id": snap_id,
                        "text_chunk": text_chunk,
                        "severity": severity,
                        "created_at": str(created_at),
                        "similarity": round(sim, 4),
                        "decay_factor": round(decay_factor, 4),
                        "score": final_score
                    })
            # Sort descending by the decay-adjusted final score
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            vector_results = scored_candidates[:5]
        latency["vector"] = round((time.perf_counter() - t_vec_start) * 1000, 2)
        
    # 3. Graph Read
    if use_graph:
        t_graph_start = time.perf_counter()
        cursor.execute("SELECT node_id, relationship FROM patient_nodes WHERE user_id = ?;", (user_id,))
        pat_nodes = cursor.fetchall()
        
        if pat_nodes:
            node_ids = [p[0] for p in pat_nodes]
            placeholders = ",".join("?" for _ in node_ids)
            # Query connections where either source or target matches any of patient nodes
            cursor.execute(f"""
                SELECT source, target, relationship
                FROM clinical_edges
                WHERE source IN ({placeholders}) OR target IN ({placeholders});
            """, node_ids + node_ids)
            edges = cursor.fetchall()
            
            graph_results = [{
                "patient_nodes": [{"node_id": p[0], "relationship": p[1]} for p in pat_nodes],
                "clinical_connections": [{"source": e[0], "target": e[1], "relationship": e[2]} for e in edges]
            }]
        else:
            graph_results = []
        latency["graph"] = round((time.perf_counter() - t_graph_start) * 1000, 2)
        
    conn.close()
    latency["total"] = round((time.perf_counter() - t_start) * 1000, 2)
    
    return RAGRetrieveResponse(
        linear_results=linear_results,
        vector_results=vector_results,
        graph_results=graph_results,
        latency_ms=latency
    )

# ---------------------------------------------------------
# 5. FastAPI App definition
# ---------------------------------------------------------
app = FastAPI(
    title="Standalone Isolated RAG Benchmarks",
    description="Offline local SQLite-based testbed comparing Linear, Vector, and Graph architectures dynamically."
)

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "Standalone Isolated RAG Benchmarks API is running!",
        "docs_url": "/docs",
        "endpoints": ["POST /store", "POST /retrieve"]
    }

@app.post("/store", response_model=RAGStoreResponse)
def api_store(
    payload: List[RAGStoreItem],
    user_id: int = Query(9999, description="Patient user ID"),
    use_linear: bool = Query(True, description="Enable Linear storage"),
    use_vector: bool = Query(True, description="Enable Vector embedding storage"),
    use_graph: bool = Query(True, description="Enable Graph relationship mapping")
):
    try:
        return store_rag_data_local(payload, user_id, use_linear, use_vector, use_graph)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve", response_model=RAGRetrieveResponse)
def api_retrieve(
    payload: RAGRetrieveRequest,
    user_id: int = Query(9999, description="Patient user ID"),
    use_linear: bool = Query(True, description="Enable Linear retrieval"),
    use_vector: bool = Query(True, description="Enable Vector search retrieval"),
    use_graph: bool = Query(True, description="Enable Graph traversal retrieval"),
    similarity_threshold: float = Query(0.70, description="Minimum similarity score threshold")
):
    try:
        return retrieve_rag_data_local(payload, user_id, use_linear, use_vector, use_graph, similarity_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# 6. Built-in Automated Tests / Benchmarking
# ---------------------------------------------------------
def run_cli_tests():
    print("=" * 60)
    print("RUNNING OFFLINE STANDALONE RAG BENCHMARK TESTS (SQLite)")
    print("=" * 60)
    
    # Remove existing db to start fresh
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except Exception:
            pass
            
    initialize_db()
    
    user_id = 7777
    mock_vector = [round(random.uniform(-0.1, 0.1), 6) for _ in range(1536)]
    clinical_note = """
    Session date: March 2026
    Primary complaint: severe headache with high bp
    Note details: Patient reports throbbing pain, tension-type headache causes, and we suggested monitor bp and rest.
    """
    
    print("\n--- 1. Store API Benchmarks (Individual & Combined Combinations) ---")
    
    # Store A: Linear Only
    req = [RAGStoreItem(id="test-1", text_chunk=clinical_note)]
    res = store_rag_data_local(req, user_id=user_id, use_linear=True, use_vector=False, use_graph=False)
    print(f"[*] Linear Store (Only): Success. Latency: {res.latency_ms}")
    
    # Store B: Vector Only
    req = [RAGStoreItem(id="test-2", text_chunk=clinical_note, embedding=mock_vector)]
    res = store_rag_data_local(req, user_id=user_id, use_linear=False, use_vector=True, use_graph=False)
    print(f"[*] Vector Store (Only): Success. Latency: {res.latency_ms}")
    
    # Store C: Graph Only
    req = [RAGStoreItem(id="test-3", text_chunk=clinical_note, graph_nodes=["headache", "high bp", "tension-type headache"])]
    res = store_rag_data_local(req, user_id=user_id, use_linear=False, use_vector=False, use_graph=True)
    print(f"[*] Graph Store (Only): Success. Latency: {res.latency_ms}")
    
    # Store D: Vector + Graph Combination
    req = [RAGStoreItem(id="test-4", text_chunk=clinical_note, embedding=mock_vector, graph_nodes=["headache", "high bp", "tension-type headache"])]
    res = store_rag_data_local(req, user_id=user_id, use_linear=False, use_vector=True, use_graph=True)
    print(f"[*] Vector + Graph Store: Success. Latency: {res.latency_ms}")
    
    # Store E: All 3 (Linear + Vector + Graph) Combined
    req = [RAGStoreItem(id="test-5", text_chunk=clinical_note, embedding=mock_vector, graph_nodes=["headache", "high bp", "tension-type headache"])]
    res = store_rag_data_local(req, user_id=user_id, use_linear=True, use_vector=True, use_graph=True)
    print(f"[*] Linear + Vector + Graph Store: Success. Latency: {res.latency_ms}")
    
    print("\n--- 2. Retrieval API Benchmarks (Individual & Combined Combinations) ---")
    
    # Retrieve A: Linear Only
    req_ret = RAGRetrieveRequest(query="headache and high bp")
    res_ret = retrieve_rag_data_local(req_ret, user_id=user_id, use_linear=True, use_vector=False, use_graph=False)
    print(f"[*] Linear Retrieve: Found {len(res_ret.linear_results or [])} items. Latency: {res_ret.latency_ms}")
    
    # Retrieve B: Vector Only
    req_ret = RAGRetrieveRequest(query="headache and high bp", query_vector=mock_vector)
    res_ret = retrieve_rag_data_local(req_ret, user_id=user_id, use_linear=False, use_vector=True, use_graph=False)
    print(f"[*] Vector Retrieve: Found {len(res_ret.vector_results or [])} items. Latency: {res_ret.latency_ms}")
    if res_ret.vector_results:
        print(f"    Top Match Cosine Similarity: {res_ret.vector_results[0]['similarity']}")
        
    # Retrieve C: Graph Only
    req_ret = RAGRetrieveRequest(query="headache and high bp")
    res_ret = retrieve_rag_data_local(req_ret, user_id=user_id, use_linear=False, use_vector=False, use_graph=True)
    print(f"[*] Graph Retrieve: Latency: {res_ret.latency_ms}")
    if res_ret.graph_results:
        print(f"    Matched Patient Nodes: {len(res_ret.graph_results[0].get('patient_nodes', []))}")
        print(f"    Traversed Clinical Edges: {len(res_ret.graph_results[0].get('clinical_connections', []))}")
        
    # Retrieve D: Vector + Graph Combination
    req_ret = RAGRetrieveRequest(query="headache and high bp", query_vector=mock_vector)
    res_ret = retrieve_rag_data_local(req_ret, user_id=user_id, use_linear=False, use_vector=True, use_graph=True)
    print(f"[*] Vector + Graph Retrieve: Latency: {res_ret.latency_ms}")
    if res_ret.vector_results:
        print(f"    Vector Text Hits: {len(res_ret.vector_results)}")
    if res_ret.graph_results:
        print(f"    Graph Patient Nodes: {len(res_ret.graph_results[0].get('patient_nodes', []))}")
        print(f"    Graph Clinical Connections: {len(res_ret.graph_results[0].get('clinical_connections', []))}")
        print("    Clinical Traversal Paths:")
        for edge in res_ret.graph_results[0].get('clinical_connections', [])[:3]:
            print(f"      - {edge['source']} --[{edge['relationship']}]--> {edge['target']}")
            
    # Retrieve E: All 3 (Linear + Vector + Graph) Combined
    req_ret = RAGRetrieveRequest(query="headache and high bp", query_vector=mock_vector)
    res_ret = retrieve_rag_data_local(req_ret, user_id=user_id, use_linear=True, use_vector=True, use_graph=True)
    print(f"[*] All 3 Combined Retrieve: Latency: {res_ret.latency_ms}")
    print(f"    Linear Results Count: {len(res_ret.linear_results or [])}")
    print(f"    Vector Results Count: {len(res_ret.vector_results or [])}")
    print(f"    Graph Results Count: {1 if res_ret.graph_results else 0}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-tests":
        run_cli_tests()
    else:
        # Start uvicorn server on port 8090
        print("Starting isolated offline RAG benchmark API on http://127.0.0.1:8090 ...")
        uvicorn.run(app, host="127.0.0.1", port=8090)
