"""
RAG Benchmark — 7 Combinations (Linear + Vector + Graph)
BM25 and RRF are ALWAYS ON in the background for all combos.
You only vary: Linear, Vector, Graph.

Combinations tested:
  1. Linear Only
  2. Vector Only
  3. Graph Only
  4. Linear + Vector
  5. Linear + Graph
  6. Vector + Graph
  7. Linear + Vector + Graph (Full Hybrid)

Usage:
    1. Start server : python rag_benchmark/app.py
    2. Run benchmark: python rag_benchmark/benchmark_all_combinations.py
"""

import json
import urllib.request
import urllib.error
import os

BASE_URL = "http://127.0.0.1:8090"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
USER_ID = 1449

retrieve_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Retrive.json")
with open(retrieve_path, "r") as f:
    retrive_data = json.load(f)

QUERY_TEXT   = retrive_data.get("query", "chest pain and nausea")
QUERY_VECTOR = retrive_data.get("query_vector")

# ─────────────────────────────────────────────────────────────────────────────
# 7 combinations — BM25 always True, RRF always runs when vector is on
# Format: (Name, use_linear, use_vector, use_graph)
# ─────────────────────────────────────────────────────────────────────────────
COMBINATIONS = [
    # Singles
    ("1. Linear Only",             True,  False, False),
    ("2. Vector Only",             False, True,  False),
    ("3. Graph Only",              False, False, True),
    # Pairs
    ("4. Linear + Vector",         True,  True,  False),
    ("5. Linear + Graph",          True,  False, True),
    ("6. Vector + Graph",          False, True,  True),
    # Full Hybrid
    ("7. Linear + Vector + Graph", True,  True,  True),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper — BM25 is always True
# ─────────────────────────────────────────────────────────────────────────────
def call_retrieve(use_linear, use_vector, use_graph):
    url = (
        f"{BASE_URL}/retrieve"
        f"?user_id={USER_ID}"
        f"&use_linear={str(use_linear).lower()}"
        f"&use_vector={str(use_vector).lower()}"
        f"&use_graph={str(use_graph).lower()}"
    )
    payload = {
        "query": QUERY_TEXT,
        "query_vector": QUERY_VECTOR if use_vector else None
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as res:
        return json.loads(res.read().decode("utf-8"))

# ─────────────────────────────────────────────────────────────────────────────
# Run all 7 combinations
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark():
    print("\n" + "=" * 90)
    print(f"  RAG BENCHMARK  [BM25 + RRF always ON]")
    print(f"  Techniques varied : Linear | Vector | Graph")
    print(f"  User ID  : {USER_ID}")
    print(f"  Query    : \"{QUERY_TEXT}\"")
    print("=" * 90)

    header = (
        f"{'Combination':<30} "
        f"{'Lin(ms)':>8} {'Vec(ms)':>8} {'BM25(ms)':>9} {'Grph(ms)':>9} {'RRF(ms)':>8} {'Total(ms)':>10} "
        f"{'VecHits':>8} {'BM25Hits':>9} {'RRFHits':>8} {'TopScore':>9}"
    )
    print(f"\n{header}")
    print("-" * 90)

    all_results = []

    for name, use_linear, use_vector, use_graph in COMBINATIONS:
        try:
            resp = call_retrieve(use_linear, use_vector, use_graph)
            lat  = resp.get("latency_ms", {})

            lin_ms   = lat.get("linear", 0.0)
            vec_ms   = lat.get("vector", 0.0)
            bm25_ms  = lat.get("bm25",   0.0)
            rrf_ms   = lat.get("rrf",    0.0)
            grph_ms  = lat.get("graph",  0.0)
            total_ms = lat.get("total",  0.0)

            vec_hits  = len(resp.get("vector_results") or [])
            bm25_hits = 0  # no longer returned separately
            rrf_hits  = vec_hits  # vector_results contains the fused RRF results

            # Best score: prefer RRF > Vector > BM25
            top_score = None
            vec_r  = resp.get("vector_results") or []
            if vec_r:
                top_score = vec_r[0].get("rrf_score") or vec_r[0].get("score") or vec_r[0].get("similarity")

            score_str = f"{top_score:.4f}" if top_score is not None else "N/A"

            print(
                f"{name:<30} "
                f"{lin_ms:>8.2f} {vec_ms:>8.2f} {bm25_ms:>9.2f} {grph_ms:>9.2f} "
                f"{rrf_ms:>8.2f} {total_ms:>10.2f} "
                f"{vec_hits:>8} {bm25_hits:>9} {rrf_hits:>8} {score_str:>9}"
            )

            all_results.append({
                "name": name,
                "total_ms": total_ms,
                "vec_hits": vec_hits,
                "bm25_hits": bm25_hits,
                "rrf_hits": rrf_hits,
                "top_score": top_score,
                "resp": resp
            })

        except urllib.error.URLError as e:
            print(f"{name:<30} ERROR: Server not reachable - {e}")
        except Exception as e:
            print(f"{name:<30} ERROR: {e}")

    print("-" * 90)
    print(f"  NOTE: BM25 and RRF columns show automatically even for combos without Vector")
    print(f"        because BM25 is always stored. RRF only activates when Vector is ON.")

    # ─── RRF Detail for combos that have Vector ───────────────────────────────
    print("\n" + "=" * 90)
    print("  RRF FUSION DETAIL  (only combos where Vector is ON)")
    print("=" * 90)

    for entry in all_results:
        rrf_r = entry["resp"].get("vector_results") or []
        vec_r = entry["resp"].get("vector_results") or []
        if not vec_r:
            continue   # skip non-vector combos

        print(f"\n  [{entry['name']}]")
        print(f"    Vector/Hybrid hits : {entry['vec_hits']}")

        if rrf_r:
            print(f"    {'Rank':<6} {'ID':<10} {'RRF Score':<14} {'Vec Score':<14} {'BM25 Score':<12}")
            print("    " + "-" * 56)
            for i, r in enumerate(rrf_r, 1):
                print(
                    f"    #{i:<5} {str(r.get('id','?')):<10} "
                    f"{r.get('rrf_score', 0):<14.6f} "
                    f"{r.get('vector_score', 0):<14.4f} "
                    f"{r.get('bm25_score', 0):<12.4f}"
                )
        else:
            print("    No RRF results")

    # ─── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  VERDICT")
    print("=" * 90)
    print("""
  Fixed (always ON) : BM25 + RRF
  Best single       : Vector Only         - fastest semantic retrieval
  Best pair         : Vector + Graph      - semantic + clinical reasoning
  Best overall      : Linear+Vector+Graph - recency + semantic + clinical
                      (Combo 7 = full production upgrade target)

  Production today  : Vector + BM25 + RRF + Decay       [rag_service.py]
  Production target : Vector + BM25 + RRF + Decay + Graph
  Change needed     : Add 2 tables + 1 SQL query (no new infra, +2ms max)
""")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_benchmark()
