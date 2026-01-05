"""
Neural Calibrator: Probability Calibration for Confidence Scores

Provides calibrated confidence estimates using isotonic regression
and feature-based approaches.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

from .models import HaltClaim, EvidenceLink, NLILabel, SourceTier, TIER_RELIABILITY
from .halt_pipeline import UncertaintyCalibrator

logger = logging.getLogger(__name__)


class IsotonicCalibrator(UncertaintyCalibrator):
    """
    Calibrator using isotonic regression.
    
    Learns a monotonic mapping from raw confidence to calibrated probability.
    Requires training data of (raw_confidence, ground_truth) pairs.
    """
    
    def __init__(self):
        """Initialize isotonic calibrator."""
        self._calibrator = None
        self._is_fitted = False
        
    def fit(self, raw_confidences: List[float], ground_truths: List[bool]):
        """
        Fit the calibrator on training data.
        
        Args:
            raw_confidences: List of raw confidence scores (0-1)
            ground_truths: List of whether claims were actually true (0 or 1)
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            
            self._calibrator = IsotonicRegression(out_of_bounds='clip')
            self._calibrator.fit(raw_confidences, ground_truths)
            self._is_fitted = True
            
        except ImportError:
            raise ImportError(
                "scikit-learn is required for IsotonicCalibrator. "
                "Install with: pip install scikit-learn"
            )
    
    def calibrate(self, claim: HaltClaim, links: List[EvidenceLink]) -> float:
        """
        Return calibrated confidence for a claim.
        
        If not fitted, returns a heuristic-based confidence.
        """
        # Compute raw confidence from links
        raw_confidence = self._compute_raw_confidence(claim, links)
        
        if self._is_fitted and self._calibrator is not None:
            # Apply isotonic calibration
            calibrated = self._calibrator.predict([raw_confidence])[0]
            return float(np.clip(calibrated, 0.0, 1.0))
        else:
            # Return raw confidence if not fitted
            return raw_confidence
    
    def _compute_raw_confidence(
        self, 
        claim: HaltClaim, 
        links: List[EvidenceLink]
    ) -> float:
        """Compute raw confidence from evidence links."""
        if not links:
            return 0.0
        
        # Weight by NLI probability and support strength
        supporting = [l for l in links if l.nli_label == NLILabel.ENTAILS]
        
        if not supporting:
            return 0.0
        
        # Average support strength weighted by NLI probability
        weighted_sum = sum(l.support_strength * l.nli_probability for l in supporting)
        total_weight = sum(l.nli_probability for l in supporting)
        
        if total_weight == 0:
            return 0.0
            
        return weighted_sum / total_weight


class FeatureBasedCalibrator(UncertaintyCalibrator):
    """
    Calibrator using multiple features for confidence estimation.
    
    Features:
    - Evidence count
    - Source tier distribution
    - Support/contradiction ratio
    - Average NLI probability
    - Claim opinion (belief/disbelief/uncertainty)
    """
    
    def __init__(
        self,
        evidence_weight: float = 0.25,
        tier_weight: float = 0.25,
        agreement_weight: float = 0.30,
        probability_weight: float = 0.20
    ):
        """
        Initialize feature-based calibrator.
        
        Args:
            evidence_weight: Weight for evidence count factor
            tier_weight: Weight for source tier factor
            agreement_weight: Weight for support/contradiction ratio
            probability_weight: Weight for average NLI probability
        """
        self.evidence_weight = evidence_weight
        self.tier_weight = tier_weight
        self.agreement_weight = agreement_weight
        self.probability_weight = probability_weight
        
        # Evidence store for looking up tier info
        self._evidence_map: Dict[str, Any] = {}
    
    def set_evidence_map(self, evidence_map: Dict[str, Any]):
        """Set evidence map for tier lookups."""
        self._evidence_map = evidence_map
    
    def calibrate(self, claim: HaltClaim, links: List[EvidenceLink]) -> float:
        """
        Return calibrated confidence using multiple features.
        """
        if not links:
            return 0.0
        
        # Feature 1: Evidence count (saturates at 5)
        evidence_factor = min(len(links) / 5.0, 1.0)
        
        # Feature 2: Source tier quality
        tier_factor = self._compute_tier_factor(links)
        
        # Feature 3: Agreement ratio
        agreement_factor = self._compute_agreement_factor(links)
        
        # Feature 4: Average NLI probability
        prob_factor = sum(l.nli_probability for l in links) / len(links)
        
        # Weighted combination
        confidence = (
            self.evidence_weight * evidence_factor +
            self.tier_weight * tier_factor +
            self.agreement_weight * agreement_factor +
            self.probability_weight * prob_factor
        )
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _compute_tier_factor(self, links: List[EvidenceLink]) -> float:
        """Compute average source tier reliability."""
        if not self._evidence_map:
            return 0.8  # Default if no map
            
        reliabilities = []
        for link in links:
            ev = self._evidence_map.get(link.evidence_id)
            if ev:
                reliabilities.append(getattr(ev, 'reliability', 0.8))
            else:
                reliabilities.append(0.8)
                
        return sum(reliabilities) / len(reliabilities) if reliabilities else 0.8
    
    def _compute_agreement_factor(self, links: List[EvidenceLink]) -> float:
        """Compute ratio of supporting vs contradicting evidence."""
        supporting = sum(1 for l in links if l.nli_label == NLILabel.ENTAILS)
        contradicting = sum(1 for l in links if l.nli_label == NLILabel.CONTRADICTS)
        
        total = supporting + contradicting
        if total == 0:
            return 0.5  # No strong evidence either way
            
        return supporting / total


class EnsembleCalibrator(UncertaintyCalibrator):
    """
    Combines multiple calibrators for robust confidence estimation.
    """
    
    def __init__(
        self,
        calibrators: Optional[List[Tuple[UncertaintyCalibrator, float]]] = None
    ):
        """
        Initialize ensemble calibrator.
        
        Args:
            calibrators: List of (calibrator, weight) tuples
        """
        if calibrators is None:
            # Default ensemble
            self._calibrators = [
                (FeatureBasedCalibrator(), 0.6),
                (IsotonicCalibrator(), 0.4)
            ]
        else:
            self._calibrators = calibrators
    
    def calibrate(self, claim: HaltClaim, links: List[EvidenceLink]) -> float:
        """Return weighted average of calibrator outputs."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for calibrator, weight in self._calibrators:
            try:
                conf = calibrator.calibrate(claim, links)
                weighted_sum += conf * weight
                total_weight += weight
            except Exception as e:
                logger.warning(f"Calibrator {type(calibrator).__name__} failed: {e}")
                continue
        
        if total_weight == 0:
            return 0.0
            
        return weighted_sum / total_weight


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def expected_calibration_error(
    confidences: List[float],
    accuracies: List[bool],
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well-calibrated a model's confidence estimates are.
    Lower is better (0 = perfectly calibrated).
    
    Args:
        confidences: List of confidence scores (0-1)
        accuracies: List of whether predictions were correct
        n_bins: Number of bins for binning confidences
        
    Returns:
        ECE score (0-1)
    """
    confidences = np.array(confidences)
    accuracies = np.array(accuracies, dtype=float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        
        if np.sum(in_bin) == 0:
            continue
            
        avg_confidence = np.mean(confidences[in_bin])
        avg_accuracy = np.mean(accuracies[in_bin])
        bin_weight = np.sum(in_bin) / len(confidences)
        
        ece += bin_weight * np.abs(avg_accuracy - avg_confidence)
    
    return float(ece)


def brier_score(confidences: List[float], outcomes: List[bool]) -> float:
    """
    Compute Brier Score.
    
    Measures the mean squared difference between predicted probabilities
    and actual outcomes. Lower is better (0 = perfect).
    
    Args:
        confidences: List of confidence scores (0-1)
        outcomes: List of actual outcomes (True/False)
        
    Returns:
        Brier score
    """
    confidences = np.array(confidences)
    outcomes = np.array(outcomes, dtype=float)
    
    return float(np.mean((confidences - outcomes) ** 2))


def reliability_diagram_data(
    confidences: List[float],
    accuracies: List[bool],
    n_bins: int = 10
) -> Dict[str, List[float]]:
    """
    Generate data for a reliability diagram.
    
    Returns:
        Dict with 'mean_confidence', 'mean_accuracy', 'count' for each bin
    """
    confidences = np.array(confidences)
    accuracies = np.array(accuracies, dtype=float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    mean_confidences = []
    mean_accuracies = []
    counts = []
    
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        
        count = np.sum(in_bin)
        counts.append(int(count))
        
        if count == 0:
            mean_confidences.append((lower + upper) / 2)
            mean_accuracies.append(0.0)
        else:
            mean_confidences.append(float(np.mean(confidences[in_bin])))
            mean_accuracies.append(float(np.mean(accuracies[in_bin])))
    
    return {
        "mean_confidence": mean_confidences,
        "mean_accuracy": mean_accuracies,
        "count": counts
    }
