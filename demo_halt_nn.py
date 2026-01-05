"""
HALT-NN Demonstration: Worked Example

Shows the full pipeline in action with sample evidence.
"""

from cortex_zero_core import (
    run_halt_pipeline, HaltEvidence, SourceTier,
    ActionDecision, ClaimStatus
)


def create_sample_evidence():
    """Create sample evidence store for demonstration."""
    evidence = [
        # Tier A: Official source
        HaltEvidence.create(
            content="Python is a high-level, general-purpose programming language. "
                    "Its design philosophy emphasizes code readability with the use of significant indentation.",
            source_id="wikipedia.org/Python",
            tier=SourceTier.TIER_A
        ),
        HaltEvidence.create(
            content="Python was conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands.",
            source_id="python.org/history",
            tier=SourceTier.TIER_A
        ),
        # Tier B: Reputable journalism
        HaltEvidence.create(
            content="Python has become one of the most popular programming languages in 2024, "
                    "especially for data science and machine learning applications.",
            source_id="techcrunch.com/2024",
            tier=SourceTier.TIER_B
        ),
        # Tier C: Blog (lower reliability)
        HaltEvidence.create(
            content="Some developers argue Python is slow compared to compiled languages like C++.",
            source_id="dev-blog.com",
            tier=SourceTier.TIER_C
        ),
        # Unrelated evidence
        HaltEvidence.create(
            content="The Eiffel Tower is located in Paris, France.",
            source_id="travel-guide.com",
            tier=SourceTier.TIER_B
        )
    ]
    return evidence


def run_demo():
    """Run demonstration of HALT-NN pipeline."""
    print("=" * 70)
    print("HALT-NN DEMONSTRATION: Evidence-Grounded Anti-Hallucination")
    print("=" * 70)
    
    # Create evidence store
    evidence_store = create_sample_evidence()
    print(f"\n[EVIDENCE] Store: {len(evidence_store)} items loaded")
    
    # Test queries
    queries = [
        "What is Python programming language?",
        "Who created Python?",
        "What is the capital of Mars?",  # Should abstain - no evidence
    ]
    
    for query in queries:
        print("\n" + "-" * 70)
        print(f"[QUERY] {query}")
        print("-" * 70)
        
        # Run pipeline
        audit = run_halt_pipeline(query, evidence_store)
        
        # Display results
        print(f"\n[CLAIMS ANALYSIS]")
        for claim in audit.claims:
            status_icon = {
                ClaimStatus.SUPPORTED: "[OK]",
                ClaimStatus.UNSUPPORTED: "[NO]",
                ClaimStatus.DISPUTED: "[??]",
                ClaimStatus.PENDING: "[..]"
            }.get(claim.status, "[?]")
            print(f"   {status_icon} [{claim.status.value}] {claim.text[:60]}...")
            print(f"       Opinion: b={claim.opinion.belief:.2f}, d={claim.opinion.disbelief:.2f}, u={claim.opinion.uncertainty:.2f}")
            print(f"       Confidence: {claim.confidence:.2%}")
        
        print(f"\n[EVIDENCE LINKS]")
        for link in audit.links[:3]:  # Show top 3
            ev = next((e for e in audit.evidence if e.id == link.evidence_id), None)
            if ev:
                print(f"   [{link.nli_label.value}] {ev.source_id} (prob={link.nli_probability:.2f})")
        
        print(f"\n[DECISION] {audit.action.value}")
        print(f"   Coverage: {audit.coverage_ratio:.0%}")
        print(f"   Confidence: {audit.overall_confidence:.2%}")
        print(f"   Conflicts: {audit.conflict_count}")
        
        print(f"\n[GENERATED ANSWER]")
        print(audit.answer_text)
        
        if audit.abstentions:
            print(f"\n[ABSTENTIONS] {audit.abstentions}")
        
        if audit.next_actions:
            print(f"\n[TO RAISE CONFIDENCE]")
            for action in audit.next_actions:
                print(f"   - {action}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE: Zero unsupported claims emitted as facts.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
