"""
NS-ARC Verifier: HALT-NN Integration for Compression Metadata Verification

Uses HALT-NN's evidence-grounded verification to validate compression claims:
- Compression ratio accuracy
- File integrity assertions
- Deduplication statistics
- Algorithm selection rationale
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class VerificationStatus(Enum):
    """Status of a verified claim."""
    VERIFIED = "verified"
    DISPUTED = "disputed"
    UNVERIFIABLE = "unverifiable"
    PENDING = "pending"


@dataclass
class CompressionClaim:
    """A claim about compression results that can be verified."""
    claim_type: str  # ratio, integrity, dedup, algorithm
    claim_text: str
    expected_value: Any
    actual_value: Optional[Any] = None
    status: VerificationStatus = VerificationStatus.PENDING
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class CompressionMetadata:
    """Metadata from NS-ARC compression that can be verified."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm_used: str
    file_type: str
    original_hash: str
    compressed_hash: str
    chunks_total: int = 0
    chunks_deduped: int = 0
    dedup_ratio: float = 0.0
    processing_time_ms: float = 0.0


class NSARCVerifier:
    """
    Verifies NS-ARC compression metadata using HALT-NN principles.
    
    Applies evidence-grounded verification to compression claims:
    1. Ratio claims: Does actual ratio match reported ratio?
    2. Integrity claims: Do hashes verify correctly?
    3. Dedup claims: Are deduplication stats accurate?
    4. Algorithm claims: Was the right algorithm selected?
    """
    
    # Tolerance for numerical comparisons
    RATIO_TOLERANCE = 0.01  # 1% tolerance
    
    # Algorithm selection rules (file_type -> expected algorithms)
    ALGORITHM_RULES = {
        "image": ["vqvae", "format_lossless", "store"],
        "video": ["residual", "store"],
        "log": ["semantic_split", "zstd"],
        "json": ["semantic_split", "zstd"],
        "text": ["bwt", "zstd"],
        "binary": ["zstd", "corpus_resonance"],
        "dna": ["dna_2bit", "zstd"],
        "audio": ["audio_psychoacoustic", "zstd"],
        "code": ["code_ast", "semantic_split", "zstd"],
    }
    
    def __init__(self):
        self.verification_history: List[Dict] = []
        
    def verify_compression(self, metadata: CompressionMetadata) -> List[CompressionClaim]:
        """
        Verify all claims in compression metadata.
        
        Returns list of verified claims with status and confidence.
        """
        claims = []
        
        # 1. Verify compression ratio claim
        claims.append(self._verify_ratio(metadata))
        
        # 2. Verify integrity claim
        claims.append(self._verify_integrity(metadata))
        
        # 3. Verify deduplication claim (if applicable)
        if metadata.chunks_total > 0:
            claims.append(self._verify_dedup(metadata))
        
        # 4. Verify algorithm selection
        claims.append(self._verify_algorithm(metadata))
        
        # Store in history
        self.verification_history.append({
            "metadata": metadata.__dict__,
            "claims": [c.__dict__ for c in claims],
            "overall_verified": all(c.status == VerificationStatus.VERIFIED for c in claims)
        })
        
        return claims
    
    def _verify_ratio(self, metadata: CompressionMetadata) -> CompressionClaim:
        """Verify that reported compression ratio is accurate."""
        claim = CompressionClaim(
            claim_type="ratio",
            claim_text=f"Compression ratio is {metadata.compression_ratio:.2f}x",
            expected_value=metadata.compression_ratio
        )
        
        # Calculate actual ratio
        if metadata.compressed_size > 0:
            actual_ratio = metadata.original_size / metadata.compressed_size
            claim.actual_value = actual_ratio
            
            # Check if within tolerance
            diff = abs(actual_ratio - metadata.compression_ratio)
            if diff <= self.RATIO_TOLERANCE * metadata.compression_ratio:
                claim.status = VerificationStatus.VERIFIED
                claim.confidence = 1.0 - (diff / metadata.compression_ratio)
                claim.evidence = [
                    f"Original size: {metadata.original_size} bytes",
                    f"Compressed size: {metadata.compressed_size} bytes",
                    f"Calculated ratio: {actual_ratio:.4f}x",
                    f"Reported ratio: {metadata.compression_ratio:.4f}x",
                    f"Difference: {diff:.4f} (within {self.RATIO_TOLERANCE*100}% tolerance)"
                ]
            else:
                claim.status = VerificationStatus.DISPUTED
                claim.confidence = max(0, 1.0 - (diff / metadata.compression_ratio))
                claim.evidence = [
                    f"MISMATCH: Calculated {actual_ratio:.4f}x != Reported {metadata.compression_ratio:.4f}x",
                    f"Difference {diff:.4f} exceeds tolerance"
                ]
        else:
            claim.status = VerificationStatus.UNVERIFIABLE
            claim.evidence = ["Compressed size is 0, cannot verify ratio"]
            
        return claim
    
    def _verify_integrity(self, metadata: CompressionMetadata) -> CompressionClaim:
        """Verify that integrity hashes are valid format."""
        claim = CompressionClaim(
            claim_type="integrity",
            claim_text="File integrity hashes are valid",
            expected_value="valid_sha256"
        )
        
        # Verify hash format (SHA-256 = 64 hex chars)
        original_valid = self._is_valid_sha256(metadata.original_hash)
        compressed_valid = self._is_valid_sha256(metadata.compressed_hash)
        
        if original_valid and compressed_valid:
            claim.status = VerificationStatus.VERIFIED
            claim.confidence = 1.0
            claim.actual_value = "both_valid"
            claim.evidence = [
                f"Original hash: {metadata.original_hash[:16]}... (valid SHA-256)",
                f"Compressed hash: {metadata.compressed_hash[:16]}... (valid SHA-256)"
            ]
        else:
            claim.status = VerificationStatus.DISPUTED
            claim.confidence = 0.5 if (original_valid or compressed_valid) else 0.0
            claim.actual_value = "invalid"
            claim.evidence = []
            if not original_valid:
                claim.evidence.append(f"Original hash invalid: {metadata.original_hash}")
            if not compressed_valid:
                claim.evidence.append(f"Compressed hash invalid: {metadata.compressed_hash}")
                
        return claim
    
    def _verify_dedup(self, metadata: CompressionMetadata) -> CompressionClaim:
        """Verify deduplication statistics are consistent."""
        claim = CompressionClaim(
            claim_type="dedup",
            claim_text=f"Deduplication ratio is {metadata.dedup_ratio:.2f}x",
            expected_value=metadata.dedup_ratio
        )
        
        # Calculate expected dedup ratio
        if metadata.chunks_total > 0:
            unique_chunks = metadata.chunks_total - metadata.chunks_deduped
            if unique_chunks > 0:
                actual_dedup = metadata.chunks_total / unique_chunks
                claim.actual_value = actual_dedup
                
                diff = abs(actual_dedup - metadata.dedup_ratio)
                if diff <= self.RATIO_TOLERANCE * max(actual_dedup, 1.0):
                    claim.status = VerificationStatus.VERIFIED
                    claim.confidence = 0.95
                    claim.evidence = [
                        f"Total chunks: {metadata.chunks_total}",
                        f"Deduplicated chunks: {metadata.chunks_deduped}",
                        f"Unique chunks: {unique_chunks}",
                        f"Calculated dedup ratio: {actual_dedup:.2f}x"
                    ]
                else:
                    claim.status = VerificationStatus.DISPUTED
                    claim.confidence = 0.5
                    claim.evidence = [
                        f"Dedup ratio mismatch: calculated {actual_dedup:.2f}x != reported {metadata.dedup_ratio:.2f}x"
                    ]
            else:
                claim.status = VerificationStatus.UNVERIFIABLE
                claim.evidence = ["All chunks deduplicated, cannot verify"]
        else:
            claim.status = VerificationStatus.UNVERIFIABLE
            claim.evidence = ["No chunk data available"]
            
        return claim
    
    def _verify_algorithm(self, metadata: CompressionMetadata) -> CompressionClaim:
        """Verify that appropriate algorithm was selected for file type."""
        claim = CompressionClaim(
            claim_type="algorithm",
            claim_text=f"Algorithm '{metadata.algorithm_used}' is appropriate for {metadata.file_type}",
            expected_value=self.ALGORITHM_RULES.get(metadata.file_type.lower(), ["zstd"])
        )
        
        expected_algos = self.ALGORITHM_RULES.get(metadata.file_type.lower(), ["zstd", "store"])
        claim.actual_value = metadata.algorithm_used.lower()
        
        if claim.actual_value in [a.lower() for a in expected_algos]:
            claim.status = VerificationStatus.VERIFIED
            claim.confidence = 1.0
            claim.evidence = [
                f"File type: {metadata.file_type}",
                f"Algorithm used: {metadata.algorithm_used}",
                f"Expected algorithms for this type: {expected_algos}",
                "✓ Algorithm selection is appropriate"
            ]
        else:
            # Not in expected list, but might still be valid
            claim.status = VerificationStatus.DISPUTED
            claim.confidence = 0.6
            claim.evidence = [
                f"File type: {metadata.file_type}",
                f"Algorithm used: {metadata.algorithm_used}",
                f"Expected: {expected_algos}",
                "⚠ Unexpected algorithm selection (may still be valid)"
            ]
            
        return claim
    
    def _is_valid_sha256(self, hash_str: str) -> bool:
        """Check if string is valid SHA-256 hash format."""
        if not hash_str or len(hash_str) != 64:
            return False
        try:
            int(hash_str, 16)
            return True
        except ValueError:
            return False
    
    def generate_verification_report(self, claims: List[CompressionClaim]) -> Dict:
        """Generate a summary verification report."""
        verified = sum(1 for c in claims if c.status == VerificationStatus.VERIFIED)
        disputed = sum(1 for c in claims if c.status == VerificationStatus.DISPUTED)
        unverifiable = sum(1 for c in claims if c.status == VerificationStatus.UNVERIFIABLE)
        
        avg_confidence = sum(c.confidence for c in claims) / len(claims) if claims else 0
        
        return {
            "total_claims": len(claims),
            "verified": verified,
            "disputed": disputed,
            "unverifiable": unverifiable,
            "overall_confidence": avg_confidence,
            "all_verified": verified == len(claims),
            "claims": [
                {
                    "type": c.claim_type,
                    "text": c.claim_text,
                    "status": c.status.value,
                    "confidence": c.confidence,
                    "evidence": c.evidence
                }
                for c in claims
            ]
        }
    
    def verify_from_json(self, json_output: str) -> Dict:
        """
        Verify compression metadata from NS-ARC JSON output.
        
        This is the main entry point for integration with NS-ARC.
        """
        try:
            data = json.loads(json_output)
            
            metadata = CompressionMetadata(
                original_size=data.get("original_size", 0),
                compressed_size=data.get("compressed_size", 0),
                compression_ratio=data.get("compression_ratio", 0.0),
                algorithm_used=data.get("algorithm", "unknown"),
                file_type=data.get("file_type", "binary"),
                original_hash=data.get("original_hash", ""),
                compressed_hash=data.get("compressed_hash", ""),
                chunks_total=data.get("chunks_total", 0),
                chunks_deduped=data.get("chunks_deduped", 0),
                dedup_ratio=data.get("dedup_ratio", 0.0),
                processing_time_ms=data.get("processing_time_ms", 0.0)
            )
            
            claims = self.verify_compression(metadata)
            return self.generate_verification_report(claims)
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid JSON: {e}",
                "total_claims": 0,
                "verified": 0,
                "overall_confidence": 0.0
            }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def verify_compression_output(json_output: str) -> Dict:
    """Quick verification of NS-ARC compression output."""
    verifier = NSARCVerifier()
    return verifier.verify_from_json(json_output)


def create_test_metadata() -> CompressionMetadata:
    """Create sample metadata for testing."""
    return CompressionMetadata(
        original_size=1000000,
        compressed_size=250000,
        compression_ratio=4.0,
        algorithm_used="zstd",
        file_type="binary",
        original_hash="a" * 64,
        compressed_hash="b" * 64,
        chunks_total=100,
        chunks_deduped=25,
        dedup_ratio=1.33
    )


if __name__ == "__main__":
    # Demo verification
    print("=" * 60)
    print("NS-ARC Verifier Demo")
    print("=" * 60)
    
    verifier = NSARCVerifier()
    metadata = create_test_metadata()
    
    print(f"\nInput Metadata:")
    print(f"  Original: {metadata.original_size:,} bytes")
    print(f"  Compressed: {metadata.compressed_size:,} bytes")
    print(f"  Ratio: {metadata.compression_ratio}x")
    print(f"  Algorithm: {metadata.algorithm_used}")
    
    claims = verifier.verify_compression(metadata)
    report = verifier.generate_verification_report(claims)
    
    print(f"\nVerification Report:")
    print(f"  Total Claims: {report['total_claims']}")
    print(f"  Verified: {report['verified']}")
    print(f"  Disputed: {report['disputed']}")
    print(f"  Confidence: {report['overall_confidence']:.1%}")
    
    print(f"\nClaim Details:")
    for claim in report['claims']:
        icon = "✓" if claim['status'] == 'verified' else "✗" if claim['status'] == 'disputed' else "?"
        print(f"  [{icon}] {claim['type']}: {claim['text']}")
        for ev in claim['evidence'][:2]:
            print(f"      - {ev}")
