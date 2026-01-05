import random
import time
import os
import json
from ns_arc_router import SemanticRouter
from ert_dedup_harness import generate_backup_corpus
from ert_harness import generate_log_corpus

def generate_code_corpus():
    """Reads the own source code files as a 'Code' dataset."""
    code_data = b""
    files = ["ns_arc_router.py", "ns_arc_neural.py", "ert_splitter.py", "ert_core.py"]
    for f in files:
        if os.path.exists(f):
            with open(f, "rb") as fh:
                code_data += fh.read() + b"\n"
    return code_data

def benchmark_integrated():
    print("--- NS-ARC INTEGRATED SYSTEM TEST ---")
    
    # 1. Initialize The Meta-Compressor
    print("\n[1] Initializing Semantic Router & Experts...")
    ns_arc = SemanticRouter()
    
    # 2. Generate Heterogeneous Data Streams
    print("\n[2] Generating Data Streams...")
    
    # Stream A: Enterprise Logs
    stream_logs = generate_log_corpus(5000)
    print(f"  > Generated LOGS stream ({len(stream_logs)/1024:.2f} KB)")
    
    # Stream B: Backup Binaries (Multi-version)
    # We join them into one stream to test if Router can handle large blobs,
    # OR we pass them one by one. The Router expects a chunk.
    # We'll pass the 2nd version which has high redundancy against the 1st
    # Note: In a real stream, we'd pass Version 1 then Version 2.
    # For this test, we just pass the redundant Version 2 to see if CDC handles it 
    # (assuming CDC index persists in the object).
    backups = generate_backup_corpus(base_size_mb=1, num_versions=2)
    # Pre-seed the first version so the 2nd has something to dedup against
    ns_arc.cdc_expert.add_stream(backups[0]) 
    stream_backup = backups[1] 
    print(f"  > Generated BACKUP stream ({len(stream_backup)/1024/1024:.2f} MB)")
    
    # Stream C: Source Code
    stream_code = generate_code_corpus()
    if len(stream_code) == 0:
        stream_code = b"def main():\n    print('Hello World')\n" * 100
    print(f"  > Generated CODE stream ({len(stream_code)/1024:.2f} KB)")
    
    # Stream D: Visual Media (Synthetic)
    from ns_arc_visual import MockImageGenerator
    raw_pixels = MockImageGenerator.generate_gradient_image(size=512)
    # fake PNG header
    stream_image = b'\x89PNG\r\n\x1a\n' + raw_pixels 
    print(f"  > Generated IMAGE stream ({len(stream_image)/1024:.2f} KB)")
    
    # Stream E: Genomics (Synthetic DNA)
    from ns_arc_genomics import MockGenomicsGenerator
    stream_dna = MockGenomicsGenerator.generate_reads(num_reads=2000)
    print(f"  > Generated DNA stream ({len(stream_dna)/1024:.2f} KB)")

    # 3. Execution Pipeline
    print("\n[3] Running Dispatch Pipeline...")
    
    test_suite = [
        ("Enterprise Logs", stream_logs),
        ("Incremental Backup", stream_backup),
        ("Python Source", stream_code),
        ("Surveillance Frame", stream_image),
        ("Human Genome Read", stream_dna)
    ]
    
    results = []
    
    for label, data in test_suite:
        t0 = time.time()
        expert_name, comp_size, ratio = ns_arc.process_stream(data)
        t1 = time.time()
        
        print(f"\n  Input: {label}")
        print(f"  -> Routed To: {expert_name}")
        print(f"  -> Estimate: {ratio:.2f}x Ratio")
        print(f"  -> Throughput: {len(data) / (t1-t0) / 1024 / 1024:.2f} MB/s")
        
        results.append({
            "type": label,
            "expert": expert_name,
            "ratio": ratio
        })

    # 4. Validation
    print("\n--- Validation Report ---")
    correct_routing = True
    
    # Check Logs -> Splitter
    if "Splitter" not in results[0]["expert"]:
        print("[FAIL] Logs routed incorrectly.")
        correct_routing = False
    
    # Check Backup -> CDC
    if "CDC" not in results[1]["expert"]:
        print("[FAIL] Backup routed incorrectly.")
        correct_routing = False
        
    # Check Code -> Neural
    if "Neural" not in results[2]["expert"]:
        print("[FAIL] Code routed incorrectly.")
        correct_routing = False
        
    # Check Image -> Visual
    if "Visual" not in results[3]["expert"]:
        print("[FAIL] Image routed incorrectly.")
        correct_routing = False
        
    # Check DNA -> GraSS
    if "GraSS" not in results[4]["expert"]:
        print("[FAIL] DNA routed incorrectly.")
        correct_routing = False
        
    if correct_routing:
        print("SUCCESS: All streams routed to optimal experts.")
        print("NS-ARC Architecture Verified.")
    else:
        print("WARNING: Routing logic needs tuning.")

if __name__ == "__main__":
    benchmark_integrated()
