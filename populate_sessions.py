import json
import os
import sys
import sqlite3

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import store_rag_data_local, RAGStoreItem, initialize_db, DB_PATH

# ---------------------------------------------------------
# 1. Local Clinical Entity Synonym Dictionary
# ---------------------------------------------------------
SYNONYM_MAP = {
    # Conditions
    "hypertension": ["hypertension", "high bp", "high blood pressure", "hypertensive", "raised bp"],
    "diabetes": ["diabetes", "diabetic", "high blood sugar", "type 2 diabetes", "hyperglycemia"],
    "asthma": ["asthma", "asthmatic", "wheezing", "reactive airway disease", "bronchospasm"],
    "allergy": ["allergy", "allergies", "allergic", "seasonal allergy", "hay fever", "allergic rhinitis"],
    "heart attack": ["heart attack", "myocardial infarction", "cardiac arrest", "coronary event", "heart failure"],
    "migraine": ["migraine", "migraines", "severe headache", "hemicrania"],
    "arthritis": ["arthritis", "arthritic", "joint inflammation", "osteoarthritis", "rheumatoid"],
    "bronchitis": ["bronchitis", "chest cold", "lung inflammation", "bronchial infection"],
    "influenza": ["influenza", "flu", "viral flu"],
    "covid-19": ["covid-19", "covid", "sars-cov-2", "corona virus"],
    "gerd": ["gerd", "acid reflux", "heartburn", "gastroesophageal reflux"],
    "gastric irritation": ["gastric irritation", "gastric mucosal irritation", "stomach irritation", "gastritis"],
    "infection": ["infection", "infectious", "underlying infection", "bacterial infection", "viral infection"],
    "neurotransmitter imbalance": ["neurotransmitter imbalance", "neurotransmitter imbalances"],
    "mental health issues": ["mental health issues", "mental illness", "emotional issues"],
    "inflammation": ["inflammation", "inflammatory", "localized inflammation"],
    "peritonitis": ["peritonitis", "peritoneal inflammation"],
    "organ dysfunction": ["organ dysfunction", "organ failure"],
    "retinal detachment": ["retinal detachment", "detached retina"],
    "eye strain": ["eye strain", "digital eye strain", "burning in the eyes"],
    "dry eye syndrome": ["dry eye syndrome", "dry eyes", "dryness in eyes"],
    "hair loss": ["hair loss", "alopecia", "shedding hair"],
    "muscle overuse": ["muscle overuse", "overuse injury"],
    "microtears": ["microtears", "muscle tears"],
    "contact dermatitis": ["contact dermatitis", "skin allergy", "skin irritation"],
    "seborrheic dermatitis": ["seborrheic dermatitis", "dandruff", "scalp irritation"],
    "vestibular dysfunction": ["vestibular dysfunction", "balance disorder", "inner ear issue"],
    "emotional dysregulation": ["emotional dysregulation", "mood swings"],
    "muscle imbalance": ["muscle imbalance", "muscle imbalances"],
    "herniated disc": ["herniated disc", "slipped disc", "disc herniation"],
    "spinal stenosis": ["spinal stenosis", "narrowing of spinal canal"],
    "vasovagal syncope": ["vasovagal syncope", "fainting episode", "syncope"],
    "gastric cancer": ["gastric cancer", "stomach cancer"],
    "trimethylaminuria": ["trimethylaminuria", "fish odor syndrome"],
    "epistaxis": ["epistaxis", "nosebleeds", "nosebleed", "bleeding from nose"],
    "hepatic failure": ["hepatic failure", "liver failure", "liver dysfunction"],
    "muscle strain": ["muscle strain", "pulled muscle", "strain"],
    "bronchoconstriction": ["bronchoconstriction", "airway narrowing", "bronchial constriction"],
    "reactive hypoglycemia": ["reactive hypoglycemia", "low blood sugar after eating"],
    "bone marrow cancer": ["bone marrow cancer", "myeloma", "leukemia"],
    "vitamin c deficiency": ["vitamin c deficiency", "scurvy", "low vitamin c"],

    # Symptoms
    "headache": ["headache", "headaches", "head pain", "cephalalgia"],
    "chest pain": ["chest pain", "angina", "chest tightness", "chest pressure"],
    "nausea": ["nausea", "nauseous", "feeling sick", "queasy", "vomiting"],
    "sweating": ["sweating", "sweaty", "diaphoresis", "perspiration", "night sweats"],
    "shortness of breath": ["shortness of breath", "sob", "difficulty breathing", "dyspnea", "breathless"],
    "joint pain": ["joint pain", "knee pain", "swelling", "joint swelling", "joint stiffness", "knee swelling"],
    "fever": ["fever", "high temperature", "febrile", "chills", "raised temperature"],
    "cough": ["cough", "coughing", "dry cough", "productive cough", "throat tickle"],
    "fatigue": ["fatigue", "tiredness", "exhaustion", "weakness", "lethargic"],
    "runny nose": ["runny nose", "rhinorrhea", "nasal discharge", "watery eyes", "nasal congestion"],
    "sneezing": ["sneezing", "sneeze", "sneezed"],
    "itching": ["itching", "itchy", "pruritus", "skin irritation", "hives"],
    "sore throat": ["sore throat", "throat pain", "pharyngitis", "pain on swallowing"],
    "dizziness": ["dizziness", "dizzy", "lightheadedness", "vertigo", "feeling faint"],
    "discomfort": ["discomfort", "stomach pain", "abdominal pain", "uneasiness"],
    "burning pain": ["burning pain", "burning discomfort"],
    "dryness": ["dryness", "dry eyes", "dry skin"],
    "balance": ["balance", "balance issues", "loss of balance"],
    "spatial orientation": ["spatial orientation", "disorientation"],
    "stress": ["stress", "stressed", "emotional stress", "mental stress"],
    "numbness": ["numbness", "numb", "loss of sensation"],
    "tingling": ["tingling", "pins and needles", "paresthesia"],

    # Treatments
    "rest": ["rest", "resting", "avoid exertion", "sleep", "bed rest"],
    "monitor bp": ["monitor bp", "blood pressure monitor", "tracking bp", "check bp"],
    "amlodipine": ["amlodipine", "norvasc", "bp pill"],
    "ibuprofen": ["ibuprofen", "advil", "motrin", "nurofen", "painkiller"],
    "paracetamol": ["paracetamol", "acetaminophen", "tylenol", "panadol"],
    "antihistamine": ["antihistamine", "claritin", "zyrtec", "allegra", "loratadine", "cetirizine"],
    "albuterol": ["albuterol", "inhaler", "ventolin"],
    "hydration": ["hydration", "drinking water", "fluids", "drink water", "fluids intake"],
    "surgical intervention": ["surgical intervention", "surgery", "operation"],
    "artificial tears": ["artificial tears", "eye drops", "lubricating drops"],
    "topical corticosteroids": ["topical corticosteroids", "steroid cream", "hydrocortisone"],
    "emollients": ["emollients", "moisturizer", "skin cream"],
    "stress management": ["stress management", "meditation", "mindfulness"],
    "bronchodilators": ["bronchodilators", "asthma medicine"],
    "insulin": ["insulin", "insulin injection"],
    "dietary supplementation": ["dietary supplementation", "vitamin supplements", "supplements"]
}

TERM_TYPES = {
    # Conditions
    "hypertension": "Condition",
    "diabetes": "Condition",
    "asthma": "Condition",
    "allergy": "Condition",
    "heart attack": "Condition",
    "migraine": "Condition",
    "arthritis": "Condition",
    "bronchitis": "Condition",
    "influenza": "Condition",
    "covid-19": "Condition",
    "gerd": "Condition",
    "gastric irritation": "Condition",
    "infection": "Condition",
    "neurotransmitter imbalance": "Condition",
    "mental health issues": "Condition",
    "inflammation": "Condition",
    "peritonitis": "Condition",
    "organ dysfunction": "Condition",
    "retinal detachment": "Condition",
    "eye strain": "Condition",
    "dry eye syndrome": "Condition",
    "hair loss": "Condition",
    "muscle overuse": "Condition",
    "microtears": "Condition",
    "contact dermatitis": "Condition",
    "seborrheic dermatitis": "Condition",
    "vestibular dysfunction": "Condition",
    "emotional dysregulation": "Condition",
    "muscle imbalance": "Condition",
    "herniated disc": "Condition",
    "spinal stenosis": "Condition",
    "vasovagal syncope": "Condition",
    "gastric cancer": "Condition",
    "trimethylaminuria": "Condition",
    "epistaxis": "Condition",
    "hepatic failure": "Condition",
    "muscle strain": "Condition",
    "bronchoconstriction": "Condition",
    "reactive hypoglycemia": "Condition",
    "bone marrow cancer": "Condition",
    "vitamin c deficiency": "Condition",

    # Symptoms
    "headache": "Symptom",
    "chest pain": "Symptom",
    "nausea": "Symptom",
    "sweating": "Symptom",
    "shortness of breath": "Symptom",
    "joint pain": "Symptom",
    "fever": "Symptom",
    "cough": "Symptom",
    "fatigue": "Symptom",
    "runny nose": "Symptom",
    "sneezing": "Symptom",
    "itching": "Symptom",
    "sore throat": "Symptom",
    "dizziness": "Symptom",
    "discomfort": "Symptom",
    "burning pain": "Symptom",
    "dryness": "Symptom",
    "balance": "Symptom",
    "spatial orientation": "Symptom",
    "stress": "Symptom",
    "numbness": "Symptom",
    "tingling": "Symptom",

    # Treatments
    "rest": "Treatment",
    "monitor bp": "Treatment",
    "amlodipine": "Treatment",
    "ibuprofen": "Treatment",
    "paracetamol": "Treatment",
    "antihistamine": "Treatment",
    "albuterol": "Treatment",
    "hydration": "Treatment",
    "surgical intervention": "Treatment",
    "artificial tears": "Treatment",
    "topical corticosteroids": "Treatment",
    "emollients": "Treatment",
    "stress management": "Treatment",
    "bronchodilators": "Treatment",
    "insulin": "Treatment",
    "dietary supplementation": "Treatment"
}

# ---------------------------------------------------------
# 2. Local Clinical Entity and Relationship Extractor
# ---------------------------------------------------------
def extract_clinical_nodes_and_edges(text: str):
    """
    Scans the text_chunk for synonym keywords.
    Returns:
        nodes: list of dicts {"id": canonical_name, "type": term_type}
        edges: list of dicts {"source": condition, "target": symptom/treatment, "relationship": rel_type}
    """
    detected_conditions = []
    detected_symptoms = []
    detected_treatments = []
    
    text_lower = text.lower()
    
    # Extract entities
    for term, synonyms in SYNONYM_MAP.items():
        for syn in synonyms:
            if syn in text_lower:
                term_type = TERM_TYPES.get(term)
                if term_type == "Condition" and term not in detected_conditions:
                    detected_conditions.append(term)
                elif term_type == "Symptom" and term not in detected_symptoms:
                    detected_symptoms.append(term)
                elif term_type == "Treatment" and term not in detected_treatments:
                    detected_treatments.append(term)
                break  # Matched one synonym, move to the next term
                
    nodes = []
    for c in detected_conditions:
        nodes.append({"id": c, "type": "Condition"})
    for s in detected_symptoms:
        nodes.append({"id": s, "type": "Symptom"})
    for t in detected_treatments:
        nodes.append({"id": t, "type": "Treatment"})
        
    edges = []
    # Dynamic Relation Mapping: Link Condition -> Symptom (causes) and Condition -> Treatment (treatment)
    for cond in detected_conditions:
        for sym in detected_symptoms:
            edges.append({"source": cond, "target": sym, "relationship": "causes"})
        for treat in detected_treatments:
            edges.append({"source": cond, "target": treat, "relationship": "treatment"})
            
    return nodes, edges

# ---------------------------------------------------------
# 3. Database seeding and population logic
# ---------------------------------------------------------
def get_db_stats():
    """Helper to return stats on the DB."""
    if not os.path.exists(DB_PATH):
        return 0, 0, 0
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM snapshots;")
        snaps = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM patient_nodes;")
        p_nodes = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM clinical_nodes;")
        c_nodes = cursor.fetchone()[0]
        conn.close()
        return snaps, p_nodes, c_nodes
    except Exception:
        return 0, 0, 0

def seed_base_vocabulary():
    """Seeds all known dictionary terms into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Insert vocabulary nodes
    for term, term_type in TERM_TYPES.items():
        cursor.execute("INSERT OR IGNORE INTO clinical_nodes (id, type) VALUES (?, ?);", (term, term_type))
    conn.commit()
    conn.close()

def populate_sessions(file_name="sessions_100.json", clear_existing=False):
    # 1. Initialize DB schema
    initialize_db()
    
    # 2. Seed basic vocabulary
    seed_base_vocabulary()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, file_name)
    
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON file '{file_name}' not found in {base_dir}")
        print("Please place your session data file there and run the script again.")
        return
        
    print(f"[INFO] Reading session data from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if not isinstance(data, list):
        print("[ERROR] JSON data must be a list of session objects.")
        return
        
    print(f"[INFO] Found {len(data)} session(s) in JSON file.")
    
    # 3. Optionally clear database
    if clear_existing:
        print("[INFO] Clearing existing snapshots, embeddings, and graph nodes to ensure a fresh import...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM snapshot_embeddings;")
        cursor.execute("DELETE FROM patient_nodes;")
        cursor.execute("DELETE FROM snapshots;")
        cursor.execute("DELETE FROM snapshots_fts;")
        cursor.execute("DELETE FROM clinical_edges;")
        cursor.execute("DELETE FROM clinical_nodes;")
        conn.commit()
        conn.close()
        print("[INFO] Database cleared.")
        # Re-seed vocabulary
        seed_base_vocabulary()
        
    initial_snaps, initial_p_nodes, initial_c_nodes = get_db_stats()
    print(f"[INFO] Initial database stats: Snapshots={initial_snaps}, Patient-Graph Links={initial_p_nodes}, Vocabulary Terms={initial_c_nodes}")
    
    # 4. Parse items into RAGStoreItem
    items_to_store = []
    for index, raw_item in enumerate(data):
        try:
            doc_id = raw_item.get("id") or raw_item.get("snapshot_id") or f"import-{index+1}"
            item = RAGStoreItem(
                id=str(doc_id),
                snapshot_id=raw_item.get("snapshot_id"),
                session_id=raw_item.get("session_id"),
                user_id=raw_item.get("user_id"),
                text_chunk=raw_item.get("text_chunk", ""),
                heal_time=raw_item.get("heal_time"),
                healing_until=raw_item.get("healing_until"),
                status=raw_item.get("status"),
                created_at=raw_item.get("created_at"),
                embedding=raw_item.get("embedding")
            )
            items_to_store.append((item, raw_item))
        except Exception as e:
            print(f"[WARNING] Failed to parse item at index {index}: {e}")
            
    if not items_to_store:
        print("[ERROR] No valid items to import.")
        return
        
    # 5. Store Linear and Vector items
    print(f"[INFO] Storing {len(items_to_store)} items in local SQLite store...")
    just_items = [x[0] for x in items_to_store]
    
    res = store_rag_data_local(
        payload=just_items,
        user_id=9999,
        use_linear=True,
        use_vector=True,
        use_graph=False  # We handle graph extraction dynamically below
    )
    
    # 6. Extract and Store Clinical Graph Nodes and Edges
    import time as time_mod
    t_graph_start = time_mod.perf_counter()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    graph_populated = False
    
    for item, raw_item in items_to_store:
        user_id = item.user_id or 9999
        text = item.text_chunk
        
        # A. Check for custom overrides in JSON
        nodes = raw_item.get("graph_nodes", [])
        edges = raw_item.get("graph_edges", [])
        
        # B. If not in JSON, extract dynamically using our rule-based system
        if not nodes and not edges:
            nodes, edges = extract_clinical_nodes_and_edges(text)
            
        if nodes or edges:
            graph_populated = True
            
            # Insert nodes
            for node in nodes:
                node_id = str(node.get("id")).strip().lower()
                node_type = str(node.get("type", "Symptom")).capitalize()
                if node_id:
                    cursor.execute("INSERT OR IGNORE INTO clinical_nodes (id, type) VALUES (?, ?);", (node_id, node_type))
                    # Link patient
                    cursor.execute("DELETE FROM patient_nodes WHERE user_id = ? AND node_id = ?;", (user_id, node_id))
                    cursor.execute("INSERT INTO patient_nodes (user_id, node_id, relationship) VALUES (?, ?, 'suffers_from');", (user_id, node_id))
            
            # Insert edges
            for edge in edges:
                src = str(edge.get("source")).strip().lower()
                tgt = str(edge.get("target")).strip().lower()
                rel = str(edge.get("relationship", "causes")).strip().lower()
                if src and tgt:
                    # Ensure node references exist
                    cursor.execute("INSERT OR IGNORE INTO clinical_nodes (id, type) VALUES (?, 'Condition');", (src,))
                    cursor.execute("INSERT OR IGNORE INTO clinical_nodes (id, type) VALUES (?, 'Symptom');", (tgt,))
                    # Insert connection
                    cursor.execute("INSERT INTO clinical_edges (source, target, relationship) VALUES (?, ?, ?);", (src, tgt, rel))
                    
    conn.commit()
    conn.close()
    
    graph_latency = round((time_mod.perf_counter() - t_graph_start) * 1000, 2)
    
    # Update latencies report to show actual graph extraction time
    latencies = dict(res.latency_ms)
    latencies["graph"] = graph_latency
    latencies["total"] = round(latencies["total"] + graph_latency, 2)
    
    # Print simplified report
    print("\n--- Storage Report ---")
    print(f"Message: {res.message}")
    print(f"Linear Stored: {res.stored_linear}")
    print(f"Vector Stored: {res.stored_vector}")
    print(f"Graph Stored:  {graph_populated}")
    print(f"Latencies (ms): {json.dumps(latencies, indent=2)}")
    print("----------------------\n")
    
    final_snaps, final_p_nodes, final_c_nodes = get_db_stats()
    print(f"[INFO] Import finished! Current database stats: Snapshots={final_snaps}, Patient-Graph Links={final_p_nodes}, Vocabulary Terms={final_c_nodes}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed database with local session JSON data.")
    parser.add_argument("--file", type=str, default="sessions_100.json", help="Name of JSON file to load.")
    parser.add_argument("--clear", action="store_true", help="Clear the database snapshots before importing.")
    
    args = parser.parse_args()
    populate_sessions(file_name=args.file, clear_existing=args.clear)
