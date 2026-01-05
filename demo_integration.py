#!/usr/bin/env python3
"""
NS-ARC + HALT-NN Integration Demo

Demonstrates how the compression system integrates with 
HALT-NN style verification to validate compression claims.
"""

from ns_arc_core import JSONLogCompressor
from ns_arc_verifier import NSARCVerifier, CompressionMetadata
import hashlib


def main():
    print("=" * 60)
    print("NS-ARC + HALT-NN Integration Demo")
    print("=" * 60)
    
    # Step 1: Create sample data (JSON logs)
    logs = [
        '{"timestamp": "2024-01-01T10:00:00", "level": "INFO", "service": "api", "msg": "Server started"}',
        '{"timestamp": "2024-01-01T10:00:01", "level": "INFO", "service": "api", "msg": "Connected to database"}',
        '{"timestamp": "2024-01-01T10:00:02", "level": "INFO", "service": "api", "msg": "Ready for connections"}',
        '{"timestamp": "2024-01-01T10:01:00", "level": "INFO", "service": "api", "msg": "Request received"}',
        '{"timestamp": "2024-01-01T10:01:01", "level": "INFO", "service": "api", "msg": "Response sent"}',
    ]
    
    # Calculate original size
    original_data = "\n".join(logs).encode("utf-8")
    original_size = len(original_data)
    original_hash = hashlib.sha256(original_data).hexdigest()
    
    print(f"\n[1] ORIGINAL DATA:")
    print(f"    Lines: {len(logs)} JSON log entries")
    print(f"    Size: {original_size} bytes")
    print(f"    SHA-256: {original_hash[:32]}...")
    
    # Step 2: Compress with NS-ARC JSONLogCompressor
    print(f"\n[2] COMPRESSING WITH NS-ARC...")
    compressor = JSONLogCompressor()
    compressed = compressor.compress_logs(logs)
    compressed_size = len(compressed)
    compressed_hash = hashlib.sha256(compressed).hexdigest()
    ratio = original_size / compressed_size
    
    print(f"    Compressed: {compressed_size} bytes")
    print(f"    Ratio: {ratio:.2f}x")
    print(f"    Saved: {((1 - compressed_size/original_size) * 100):.1f}%")
    
    # Step 3: Create metadata for verification
    print(f"\n[3] CREATING VERIFICATION METADATA...")
    metadata = CompressionMetadata(
        original_size=original_size,
        compressed_size=compressed_size,
        compression_ratio=ratio,
        algorithm_used="semantic_split",
        file_type="log",
        original_hash=original_hash,
        compressed_hash=compressed_hash,
        chunks_total=5,
        chunks_deduped=2,
        dedup_ratio=1.67
    )
    print(f"    Metadata created with 4 verifiable claims")
    
    # Step 4: Verify with HALT-NN style verification
    print(f"\n[4] VERIFYING CLAIMS WITH HALT-NN...")
    verifier = NSARCVerifier()
    claims = verifier.verify_compression(metadata)
    report = verifier.generate_verification_report(claims)
    
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    print(f"Total Claims Verified: {report['total_claims']}")
    print(f"  [OK] Verified: {report['verified']}")
    print(f"  [X]  Disputed: {report['disputed']}")
    print(f"  [?]  Unverifiable: {report['unverifiable']}")
    print(f"\nOverall Confidence: {report['overall_confidence']*100:.1f}%")
    print(f"All Claims Verified: {'YES' if report['all_verified'] else 'NO'}")
    
    print("\n--- CLAIM DETAILS ---")
    for claim in report["claims"]:
        icon = "[OK]" if claim["status"] == "verified" else "[X]" if claim["status"] == "disputed" else "[?]"
        print(f"\n{icon} {claim['type'].upper()}")
        print(f"    Claim: {claim['text']}")
        print(f"    Status: {claim['status'].upper()}")
        print(f"    Confidence: {claim['confidence']*100:.0f}%")
        if claim["evidence"]:
            print(f"    Evidence:")
            for ev in claim["evidence"][:2]:
                print(f"      - {ev}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print("\nThis demonstrates how NS-ARC compression claims are")
    print("verified using HALT-NN's evidence-grounded approach.")


if __name__ == "__main__":
    main()

