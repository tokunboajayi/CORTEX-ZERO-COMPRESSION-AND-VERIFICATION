import random
import time
import json
import zlib
import secrets
from ert_splitter import JSONSplitter

def generate_log_corpus(num_lines=10000):
    """
    Generates a synthetic log corpus with realistic redundancy.
    Simulates: HTTP Access Logs + App JSON Logs
    """
    data = []
    
    ips = [f"192.168.1.{i}" for i in range(1, 20)]
    agents = ["Mozilla/5.0", "Check_MK", "Prometheus/2.1"]
    eps = ["/api/v1/user", "/ping", "/admin", "/login"]
    
    start_ts = 1600000000
    
    print(f"Generating {num_lines} synthetic log lines...")
    
    for i in range(num_lines):
        # 80% Type A (Access Log JSON) -> Highly templated
        # 20% Type B (Metric JSON) -> Numeric heavy
        
        ts = start_ts + i * random.randint(1, 5) # Monotonic timestamp
        
        if random.random() < 0.8:
            log = {
                "ts": ts,
                "level": "INFO",
                "ip": random.choice(ips),
                "user_agent": random.choice(agents),
                "request": {
                    "method": "GET",
                    "uri": random.choice(eps),
                    "status": 200
                },
                "latency_ms": random.randint(5, 500)
            }
        else:
            log = {
                "ts": ts,
                "level": "DEBUG",
                "metric": "cpu_usage",
                "value": random.random() * 100.0,
                "host": f"server-{random.randint(1,5)}"
            }
            
        data.append(json.dumps(log))
        
    return "\n".join(data).encode('utf-8')

def benchmark():
    # 1. Create Corpus
    raw_data = generate_log_corpus(15000)
    raw_size = len(raw_data)
    
    print(f"\n--- Benchmark Config ---")
    print(f"Original Size: {raw_size/1024:.2f} KB")
    
    # 2. Baseline: Zlib (Standard DEFLATE)
    t0 = time.time()
    zlib_data = zlib.compress(raw_data, level=9)
    t1 = time.time()
    zlib_size = len(zlib_data)
    
    print(f"\n--- Baseline: Zlib -9 ---")
    print(f"Compressed Size: {zlib_size/1024:.2f} KB")
    print(f"Ratio: {raw_size/zlib_size:.2f}x")
    print(f"Time: {(t1-t0)*1000:.2f} ms")
    
    # 3. ERT Prototype (Semantic Split + Delta + Zlib)
    print(f"\n--- ERT Prototype (Semantic Splitting) ---")
    splitter = JSONSplitter()
    
    t0 = time.time()
    # Ingest line by line
    for line in raw_data.decode('utf-8').split('\n'):
        if line:
            splitter.ingest(line)
            
    # Compress decomposed streams
    ert_size = splitter.compress_streams()
    t1 = time.time()
    
    print(f"Stats: {splitter.get_stats()}")
    print(f"Compressed Size: {ert_size/1024:.2f} KB")
    print(f"Ratio: {raw_size/ert_size:.2f}x")
    print(f"Time: {(t1-t0)*1000:.2f} ms")
    
    # 4. Conclusion
    improvement = ((zlib_size - ert_size) / zlib_size) * 100
    print(f"\n>>> Improvement over Zlib: {improvement:.2f}%")
    if improvement > 10:
        print("RESULT: SUCCESS. Separation Hypothesis Confirmed.")
    else:
        print("RESULT: FAIL. Overhead exceeds gains.")

if __name__ == "__main__":
    benchmark()
