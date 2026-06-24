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
    id: Optional[Any] = None
    snapshot_id: Optional[int] = None
    session_id: Optional[int] = None
    user_id: Optional[int] = None
    text_chunk: str
    heal_time: Optional[int] = None
    healing_until: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
    embedding: Optional[List[float]] = None

class RAGStoreRequest(BaseModel):
    items: List[RAGStoreItem]

class RAGStoreResponse(BaseModel):
    message: str
    stored_linear: bool
    stored_vector: bool
    stored_bm25: bool
    stored_graph: bool
    latency_ms: Dict[str, float]

class RAGRetrieveRequest(BaseModel):
    query: str
    query_vector: Optional[List[float]] = None

class RAGRetrieveResponse(BaseModel):
    linear_results: Optional[List[Dict[str, Any]]] = None
    vector_results: Optional[List[Dict[str, Any]]] = None
    graph_results: Optional[List[Dict[str, Any]]] = None
    formatted_llm_context: Optional[str] = None  # Formatted text context for LLM prompt
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
    
    # Drop existing tables if the schema is old (missing 'severity' or composite keys)
    try:
        cursor.execute("PRAGMA table_info(snapshots);")
        columns = [r[1] for r in cursor.fetchall()]
        cursor.execute("PRAGMA table_info(snapshot_embeddings);")
        emb_columns = [r[1] for r in cursor.fetchall()]
        
        if (columns and "heal_time" not in columns) or (columns and "severity" not in columns) or (emb_columns and "user_id" not in emb_columns):
            cursor.execute("DROP TABLE IF EXISTS snapshot_embeddings;")
            cursor.execute("DROP TABLE IF EXISTS patient_nodes;")
            cursor.execute("DROP TABLE IF EXISTS snapshots;")
            cursor.execute("DROP TABLE IF EXISTS snapshots_fts;")
    except Exception:
        pass
    
    # 1. Snapshots (Linear Store)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id TEXT NOT NULL,
            snapshot_id INTEGER,
            user_id INTEGER NOT NULL,
            session_id INTEGER,
            text_chunk TEXT NOT NULL,
            severity TEXT,
            heal_time INTEGER,
            healing_until TEXT,
            status TEXT,
            created_at TIMESTAMP,
            PRIMARY KEY (id, user_id)
        );
    """)
    
    # 2. Vector Store (Stored as JSON string representation)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshot_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            embedding TEXT NOT NULL,
            FOREIGN KEY (snapshot_id, user_id) REFERENCES snapshots (id, user_id) ON DELETE CASCADE
        );
    """)

    # 2b. BM25 Full-Text Search (SQLite FTS5 — equivalent of PostgreSQL ts_rank_cd)
    # Porter stemming tokenizer: 'chest pain' also matches 'chest painful', 'pain' etc.
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS snapshots_fts USING fts5(
            snapshot_id UNINDEXED,
            user_id UNINDEXED,
            text_chunk,
            tokenize = 'porter ascii'
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

# Helper: RRF (Reciprocal Rank Fusion) — same algorithm as production rag_service.py
# Merges vector ranked list + BM25 ranked list into a single fused ranking.
# k=60 is standard (same as production). Higher k = less aggressive fusion.
def rrf_fuse(vector_results: list, bm25_results: list, k: int = 60) -> list:
    from collections import defaultdict
    scores = defaultdict(lambda: {"rrf": 0.0, "vector_score": 0.0, "bm25_score": 0.0, "data": None})

    for rank, item in enumerate(vector_results, 1):
        doc_id = item["id"]
        scores[doc_id]["rrf"] += 1.0 / (k + rank)
        scores[doc_id]["vector_score"] = item.get("similarity", 0.0)
        scores[doc_id]["data"] = item

    for rank, item in enumerate(bm25_results, 1):
        doc_id = item["id"]
        scores[doc_id]["rrf"] += 1.0 / (k + rank)
        scores[doc_id]["bm25_score"] = item.get("bm25_score", 0.0)
        if scores[doc_id]["data"] is None:
            scores[doc_id]["data"] = item

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x]["rrf"], reverse=True)
    result = []
    seen_sessions = set()
    for doc_id in sorted_ids:
        s = scores[doc_id]
        item_data = s["data"]
        sid = item_data.get("session_id")
        if sid not in seen_sessions:
            seen_sessions.add(sid)
            merged = {**item_data}
            merged["rrf_score"]    = round(s["rrf"], 6)
            merged["vector_score"] = round(s["vector_score"], 4)
            merged["bm25_score"]   = round(s["bm25_score"], 4)
            
            # Populate missing production schema fields
            text_chunk = merged.get("text_chunk", "")
            heal_time = merged.get("heal_time", 14)
            merged["primary_complaint"] = text_chunk
            merged["symptom_duration"] = f"{heal_time or 14} days"
            merged["symptom_tags"] = [line.strip("- ") for line in text_chunk.split("\n") if line.strip().startswith("-")] or [text_chunk[:100]]
            merged["summary"] = {
                "clinical_note": text_chunk,
                "narrative": text_chunk,
                "top_hypothesis": "gastritis or gastric irritation",
                "sections": {
                    "what_to_do_right_now": {
                        "steps": [
                            "Drink water to stay hydrated.",
                            "Avoid spicy and oily food.",
                            "Rest and monitor symptoms."
                        ]
                    }
                }
            }
            merged["relevant_messages"] = []
            
            result.append(merged)
            if len(result) >= 3:
                break
    return result

# ---------------------------------------------------------
# 3. Store Implementation
# ---------------------------------------------------------
def store_rag_data_local(
    payload: List[RAGStoreItem],
    user_id: int,
    use_linear: bool,
    use_vector: bool,
    use_graph: bool,
    use_bm25: bool = True   # BM25 is auto-stored alongside linear (no extra cost)
) -> RAGStoreResponse:
    initialize_db()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    latency = {"linear": 0.0, "vector": 0.0, "bm25": 0.0, "graph": 0.0, "total": 0.0}
    stored_linear = False
    stored_vector = False
    stored_bm25   = False
    stored_graph  = False
    
    t_start = time.perf_counter()
    
    for item in payload:
        snapshot_id = str(item.id or item.snapshot_id)
        item_user_id = item.user_id if item.user_id is not None else user_id
        
        # Automatically extract severity and created_at from text_chunk standard lines or properties
        severity = "moderate"
        text_lower = item.text_chunk.lower()
        if "severe" in text_lower:
            severity = "severe"
        elif "mild" in text_lower:
            severity = "mild"
            
        created_at = item.created_at
        if not created_at:
            for line in item.text_chunk.split("\n"):
                line_lower = line.lower().strip()
                if line_lower.startswith("severity:"):
                    severity = line.split(":", 1)[1].strip().lower()
                elif line_lower.startswith("session date:"):
                    date_str = line.split(":", 1)[1].strip()
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(date_str, "%B %Y")
                        # Store as ISO format
                        created_at = dt.strftime("%Y-%m-%dT%H:%M:%S.000000+00:00")
                    except Exception:
                        pass
        
        if not created_at:
            from datetime import datetime, timezone
            created_at = datetime.now(timezone.utc).isoformat()
            
        # 1. Base snapshot (Linear Store / Relational)
        # INSERT OR IGNORE — once stored, a snapshot is permanent and never overwritten.
        # If the same (id, user_id) is submitted again it is silently skipped.
        t_linear_start = time.perf_counter()
        if use_linear or use_vector or use_graph:
            cursor.execute(
                """
                INSERT OR IGNORE INTO snapshots (id, snapshot_id, user_id, session_id, text_chunk, severity, heal_time, healing_until, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    snapshot_id, 
                    item.snapshot_id, 
                    item_user_id, 
                    item.session_id, 
                    item.text_chunk, 
                    severity, 
                    item.heal_time, 
                    item.healing_until, 
                    item.status, 
                    created_at
                )
            )
            stored_linear = use_linear
            latency["linear"] += (time.perf_counter() - t_linear_start) * 1000
            
        # 2. Vector Store
        # INSERT OR IGNORE — skip silently if embedding already exists for this snapshot.
        if use_vector:
            t_vec_start = time.perf_counter()
            if item.embedding:
                cursor.execute(
                    "INSERT OR IGNORE INTO snapshot_embeddings (snapshot_id, user_id, embedding) VALUES (?, ?, ?);",
                    (snapshot_id, item_user_id, json.dumps(item.embedding))
                )
                stored_vector = True
            latency["vector"] += (time.perf_counter() - t_vec_start) * 1000
            
        # 3. BM25 Full-Text Store (FTS5)
        # Only insert if this snapshot is not already in the FTS index.
        if use_bm25 or use_linear:
            t_bm25_start = time.perf_counter()
            cursor.execute(
                "SELECT COUNT(*) FROM snapshots_fts WHERE snapshot_id = ? AND user_id = ?;",
                (snapshot_id, item_user_id)
            )
            already_exists = cursor.fetchone()[0] > 0
            if not already_exists:
                cursor.execute(
                    "INSERT INTO snapshots_fts (snapshot_id, user_id, text_chunk) VALUES (?, ?, ?);",
                    (snapshot_id, item_user_id, item.text_chunk)
                )
            stored_bm25 = True
            latency["bm25"] += (time.perf_counter() - t_bm25_start) * 1000

        # 4. Graph Store
        if use_graph:
            t_graph_start = time.perf_counter()
            matched_nodes = []
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
    latency["bm25"]   = round(latency["bm25"],   2)
    latency["graph"]  = round(latency["graph"],  2)
    latency["total"]  = round((time.perf_counter() - t_start) * 1000, 2)
    
    return RAGStoreResponse(
        message=f"Successfully stored {len(payload)} item(s) in local SQLite database.",
        stored_linear=stored_linear,
        stored_vector=stored_vector,
        stored_bm25=stored_bm25,
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
    use_bm25: bool = True,
    similarity_threshold: float = 0.35
) -> RAGRetrieveResponse:
    initialize_db()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    latency = {"linear": 0.0, "vector": 0.0, "bm25": 0.0, "rrf": 0.0, "graph": 0.0, "total": 0.0}
    linear_results = None
    vector_results = None
    bm25_results   = None
    rrf_results    = None
    graph_results  = None
    
    t_start = time.perf_counter()
    
    # 1. Linear Read
    if use_linear:
        t_lin_start = time.perf_counter()
        cursor.execute(
            "SELECT id, snapshot_id, session_id, text_chunk, severity, heal_time, healing_until, status, created_at FROM snapshots WHERE user_id = ? ORDER BY created_at DESC LIMIT 3;",
            (user_id,)
        )
        linear_results = []
        for r in cursor.fetchall():
            text_chunk = r[3]
            heal_time = r[5]
            linear_results.append({
                "id": r[0],
                "snapshot_id": r[1],
                "session_id": r[2],
                "text_chunk": text_chunk,
                "severity": r[4],
                "heal_time": heal_time,
                "healing_until": r[6],
                "status": r[7],
                "created_at": str(r[8]),
                # Populate missing production schema fields
                "primary_complaint": text_chunk,
                "symptom_duration": f"{heal_time or 14} days",
                "symptom_tags": [line.strip("- ") for line in text_chunk.split("\n") if line.strip().startswith("-")] or [text_chunk[:100]],
                "summary": {
                    "clinical_note": text_chunk,
                    "narrative": text_chunk,
                    "top_hypothesis": "gastritis or gastric irritation",
                    "sections": {
                        "what_to_do_right_now": {
                            "steps": [
                                "Drink water to stay hydrated.",
                                "Avoid spicy and oily food.",
                                "Rest and monitor symptoms."
                            ]
                        }
                    }
                },
                "relevant_messages": []
            })
        latency["linear"] = round((time.perf_counter() - t_lin_start) * 1000, 2)
        
    # 2. Vector Read (In-memory Cosine Similarity with Similarity Thresholding & Time Decay)
    if use_vector:
        t_vec_start = time.perf_counter()
        if payload.query_vector:
            cursor.execute("""
                SELECT s.id, s.snapshot_id, s.session_id, s.text_chunk, s.severity, s.heal_time, s.healing_until, s.status, s.created_at, e.embedding
                FROM snapshot_embeddings e
                JOIN snapshots s ON e.snapshot_id = s.id AND e.user_id = s.user_id
                WHERE s.user_id = ?;
            """, (user_id,))
            candidates = cursor.fetchall()
            
            scored_candidates = []
            for id_val, snap_id, session_id, text_chunk, severity, heal_time, healing_until, status, created_at, emb_str in candidates:
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
                        "session_id": session_id,
                        "text_chunk": text_chunk,
                        "severity": severity,
                        "heal_time": heal_time,
                        "healing_until": healing_until,
                        "status": status,
                        "created_at": str(created_at),
                        "similarity": round(sim, 4),
                        "decay_factor": round(decay_factor, 4),
                        "score": final_score
                    })
            # Sort descending by the decay-adjusted final score
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # Deduplicate by session_id to return distinct sessions
            seen_sessions = set()
            unique_candidates = []
            for cand in scored_candidates:
                sid = cand["session_id"]
                if sid not in seen_sessions:
                    seen_sessions.add(sid)
                    text_chunk = cand["text_chunk"]
                    heal_time = cand["heal_time"]
                    cand_with_fields = {
                        **cand,
                        "primary_complaint": text_chunk,
                        "symptom_duration": f"{heal_time or 14} days",
                        "symptom_tags": [line.strip("- ") for line in text_chunk.split("\n") if line.strip().startswith("-")] or [text_chunk[:100]],
                        "summary": {
                            "clinical_note": text_chunk,
                            "narrative": text_chunk,
                            "top_hypothesis": "gastritis or gastric irritation",
                            "sections": {
                                "what_to_do_right_now": {
                                    "steps": [
                                        "Drink water to stay hydrated.",
                                        "Avoid spicy and oily food.",
                                        "Rest and monitor symptoms."
                                    ]
                                }
                            }
                        },
                        "relevant_messages": []
                    }
                    unique_candidates.append(cand_with_fields)
            vector_results = unique_candidates[:3]
        latency["vector"] = round((time.perf_counter() - t_vec_start) * 1000, 2)
        
    # 3. BM25 Read (SQLite FTS5 keyword search with Porter stemming)
    if use_bm25:
        t_bm25_start = time.perf_counter()
        try:
            # Sanitize query — remove non-alpha tokens to avoid FTS5 syntax errors
            clean_tokens = [t for t in payload.query.split() if t.isalpha()]
            fts_query = " ".join(clean_tokens) if clean_tokens else payload.query

            cursor.execute("""
                SELECT
                    f.snapshot_id,
                    f.user_id,
                    s.snapshot_id  AS snap_id_int,
                    s.text_chunk,
                    s.severity,
                    s.created_at,
                    s.session_id,
                    s.heal_time,
                    s.healing_until,
                    s.status,
                    -bm25(snapshots_fts) AS bm25_score
                FROM snapshots_fts f
                JOIN snapshots s ON s.id = f.snapshot_id AND s.user_id = f.user_id
                WHERE snapshots_fts MATCH ?
                  AND f.user_id = ?
                ORDER BY bm25_score DESC
                LIMIT 3;
            """, (fts_query, user_id))
            rows = cursor.fetchall()
            bm25_results = [
                {
                    "id":          r[0],
                    "snapshot_id": r[2],
                    "text_chunk":  r[3],
                    "severity":    r[4],
                    "created_at":  str(r[5]),
                    "session_id":  r[6],
                    "heal_time":   r[7],
                    "healing_until": r[8],
                    "status":      r[9],
                    "bm25_score":  round(r[10], 4),
                    # Populate missing production fields
                    "primary_complaint": r[3],
                    "symptom_duration": f"{r[7] or 14} days",
                    "symptom_tags": [line.strip("- ") for line in r[3].split("\n") if line.strip().startswith("-")] or [r[3][:100]],
                    "summary": {
                        "clinical_note": r[3],
                        "narrative": r[3],
                        "top_hypothesis": "gastritis or gastric irritation",
                        "sections": {
                            "what_to_do_right_now": {
                                "steps": [
                                    "Drink water to stay hydrated.",
                                    "Avoid spicy and oily food.",
                                    "Rest and monitor symptoms."
                                ]
                            }
                        }
                    },
                    "relevant_messages": []
                }
                for r in rows
            ]
        except Exception as e:
            bm25_results = [{"error": str(e)}]
        latency["bm25"] = round((time.perf_counter() - t_bm25_start) * 1000, 2)

    # 3b. RRF Fusion — mirrors production rag_service.py FULL OUTER JOIN behavior.
    # Runs when EITHER side has results (not both required).
    # If BM25 returns 0 hits, vector results still get RRF scores (and vice versa).
    # Only skips if both are empty.
    if use_vector and use_bm25 and (vector_results or bm25_results):
        t_rrf_start = time.perf_counter()
        rrf_results = rrf_fuse(vector_results or [], bm25_results or [])
        latency["rrf"] = round((time.perf_counter() - t_rrf_start) * 1000, 2)
        # Overwrite vector_results with RRF fused results as the final hybrid search results
        vector_results = rrf_results


    # 4. Graph Read (Query-Focused Traversal)
    if use_graph:
        t_graph_start = time.perf_counter()
        
        # A. Extract search terms from query and find matching vocabulary in DB
        query_words = [w.strip(",.?()[]").lower() for w in payload.query.split() if len(w) > 2]
        cursor.execute("SELECT id FROM clinical_nodes;")
        all_vocab_ids = [r[0] for r in cursor.fetchall()]
        
        query_node_ids = []
        for word in query_words:
            for vid in all_vocab_ids:
                if word in vid.lower() or vid.lower() in word:
                    if vid not in query_node_ids:
                        query_node_ids.append(vid)
                        
        # B. Get all patient graph nodes
        cursor.execute("SELECT node_id, relationship FROM patient_nodes WHERE user_id = ?;", (user_id,))
        patient_links = {r[0]: r[1] for r in cursor.fetchall()}
        
        if query_node_ids and patient_links:
            # C. Fetch edges connected to matched query terms
            placeholders = ",".join("?" for _ in query_node_ids)
            cursor.execute(f"""
                SELECT source, target, relationship
                FROM clinical_edges
                WHERE source IN ({placeholders}) OR target IN ({placeholders});
            """, query_node_ids + query_node_ids)
            edges = cursor.fetchall()
            
            # D. Filter to only include nodes/edges that the patient actually has
            filtered_edges = []
            matched_nodes = set(query_node_ids)
            for src, tgt, rel in edges:
                src_ok = (src in patient_links) or (src in query_node_ids)
                tgt_ok = (tgt in patient_links) or (tgt in query_node_ids)
                if src_ok and tgt_ok:
                    filtered_edges.append({"source": src, "target": tgt, "relationship": rel})
                    matched_nodes.add(src)
                    matched_nodes.add(tgt)
                    
            # Keep patient links that are part of this query subgraph
            filtered_patient_nodes = [
                {"node_id": nid, "relationship": patient_links.get(nid, "suffers_from")}
                for nid in matched_nodes if nid in patient_links
            ]
            
            graph_results = [{
                "patient_nodes": filtered_patient_nodes,
                "clinical_connections": filtered_edges
            }]
        else:
            graph_results = []
            
        latency["graph"] = round((time.perf_counter() - t_graph_start) * 1000, 2)
        
    # Format unified context to feed directly into the LLM
    formatted_parts = []
    
    # 1. Timeline History (Linear)
    if use_linear and linear_results:
        timeline_txt = ["=== Patient History Timeline ==="]
        for r in linear_results:
            timeline_txt.append(
                f"- Session #{r.get('session_id') or 'N/A'} (Date: {r.get('created_at') or 'Unknown'}, Status: {r.get('status') or 'unknown'}):\n"
                f"  Note: {r.get('text_chunk').strip()}\n"
                f"  Heal Time: {r.get('heal_time') or 'N/A'} days (Healing until: {r.get('healing_until') or 'N/A'})"
            )
        formatted_parts.append("\n".join(timeline_txt))
        
    # 2. Similar Cases (Vector / RRF)
    # Use RRF fused if both vector and bm25 were active, otherwise vector
    similar_cases = rrf_results if (use_vector and use_bm25 and rrf_results) else (vector_results if use_vector else None)
    if similar_cases:
        cases_txt = ["=== Similar Cases (Semantic Search) ==="]
        for r in similar_cases[:3]:
            score_type = "RRF Score" if (use_vector and use_bm25 and rrf_results) else "Similarity"
            score_val = r.get("rrf_score") or r.get("score") or r.get("similarity") or 0.0
            cases_txt.append(
                f"- Case #{r.get('snapshot_id') or 'N/A'} ({score_type}: {score_val}, Status: {r.get('status') or 'unknown'}):\n"
                f"  Note: {r.get('text_chunk').strip()}"
            )
        formatted_parts.append("\n".join(cases_txt))
        
    # 3. Clinical Knowledge Graph
    if use_graph and graph_results and len(graph_results) > 0:
        g = graph_results[0]
        if g.get("patient_nodes") or g.get("clinical_connections"):
            graph_txt = ["=== Clinical Knowledge Graph Context ==="]
            if g.get("patient_nodes"):
                graph_txt.append("* Patient Symptoms & Conditions:")
                for n in g["patient_nodes"]:
                    graph_txt.append(f"  - {n['node_id']} (Relationship: {n['relationship']})")
            if g.get("clinical_connections"):
                graph_txt.append("* Clinical Relationships Map:")
                for c in g["clinical_connections"]:
                    graph_txt.append(f"  - {c['source']} --[{c['relationship']}]--> {c['target']}")
            formatted_parts.append("\n".join(graph_txt))
            
    formatted_llm_context = "\n\n".join(formatted_parts) if formatted_parts else None

    conn.close()
    latency["total"] = round((time.perf_counter() - t_start) * 1000, 2)
    
    return RAGRetrieveResponse(
        linear_results=linear_results,
        vector_results=vector_results,
        graph_results=graph_results,
        formatted_llm_context=formatted_llm_context,
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
        "endpoints": ["POST /store", "POST /retrieve", "GET /stats"]
    }

@app.get("/stats")
def get_stats():
    initialize_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get count of snapshots per user_id
    cursor.execute("SELECT user_id, count(*) FROM snapshots GROUP BY user_id;")
    snapshots_counts = {str(row[0]): row[1] for row in cursor.fetchall()}
    
    # Get count of embeddings per user_id
    cursor.execute("""
        SELECT s.user_id, count(e.id) 
        FROM snapshots s 
        LEFT JOIN snapshot_embeddings e ON s.id = e.snapshot_id AND s.user_id = e.user_id
        GROUP BY s.user_id;
    """)
    embeddings_counts = {str(row[0]): row[1] for row in cursor.fetchall()}
    
    # Get count of patient graph nodes per user_id
    cursor.execute("SELECT user_id, count(*) FROM patient_nodes GROUP BY user_id;")
    patient_nodes_counts = {str(row[0]): row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    # Combine stats per user
    all_users = set(list(snapshots_counts.keys()) + list(embeddings_counts.keys()) + list(patient_nodes_counts.keys()))
    user_stats = {}
    for uid in all_users:
        user_stats[uid] = {
            "snapshots_count": snapshots_counts.get(uid, 0),
            "embeddings_count": embeddings_counts.get(uid, 0),
            "graph_patient_nodes_count": patient_nodes_counts.get(uid, 0)
        }
        
    return {
        "database_path": DB_PATH,
        "total_snapshots": sum(snapshots_counts.values()),
        "users": user_stats
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

@app.post("/retrieve", response_model=RAGRetrieveResponse, response_model_exclude_none=True)
def api_retrieve(
    payload: RAGRetrieveRequest,
    user_id: int = Query(9999, description="Patient user ID"),
    use_linear: bool = Query(True, description="Enable Linear retrieval"),
    use_vector: bool = Query(True, description="Enable Vector search retrieval"),
    use_graph: bool = Query(True, description="Enable Graph traversal retrieval"),
    similarity_threshold: float = Query(0.35, description="Minimum similarity score threshold for vector")
):
    try:
        return retrieve_rag_data_local(
            payload=payload,
            user_id=user_id,
            use_linear=use_linear,
            use_vector=use_vector,
            use_graph=use_graph,
            use_bm25=True,
            similarity_threshold=similarity_threshold
        )
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
    req = [RAGStoreItem(id="test-3", text_chunk=clinical_note)]
    res = store_rag_data_local(req, user_id=user_id, use_linear=False, use_vector=False, use_graph=True)
    print(f"[*] Graph Store (Only): Success. Latency: {res.latency_ms}")
    
    # Store D: Vector + Graph Combination
    req = [RAGStoreItem(id="test-4", text_chunk=clinical_note, embedding=mock_vector)]
    res = store_rag_data_local(req, user_id=user_id, use_linear=False, use_vector=True, use_graph=True)
    print(f"[*] Vector + Graph Store: Success. Latency: {res.latency_ms}")
    
    # Store E: All 3 (Linear + Vector + Graph) Combined
    req = [RAGStoreItem(id="test-5", text_chunk=clinical_note, embedding=mock_vector)]
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
        print(f"    Top Match Cosine Similarity: {res_ret.vector_results[0].get('similarity') or res_ret.vector_results[0].get('vector_score', 0.0)}")
        
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
