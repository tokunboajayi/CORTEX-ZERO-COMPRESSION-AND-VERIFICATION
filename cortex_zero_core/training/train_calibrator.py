"""
Calibrator Training: Train probability calibration models

Provides tools for training and evaluating calibrators for confidence estimation.
"""

from typing import List, Tuple, Optional, Dict, Any
import logging
import json
import os
import pickle

logger = logging.getLogger(__name__)


class CalibratorTrainer:
    """
    Train calibration models on historical verification data.
    
    Uses raw confidence scores and ground truth labels to learn
    a monotonic mapping that produces well-calibrated probabilities.
    """
    
    def __init__(
        self,
        output_dir: str = "./models/calibrator",
        method: str = "isotonic"  # "isotonic" or "platt"
    ):
        """
        Initialize calibrator trainer.
        
        Args:
            output_dir: Directory to save trained calibrator
            method: Calibration method ("isotonic" or "platt")
        """
        self.output_dir = output_dir
        self.method = method
        self._calibrator = None
        
    def train(
        self,
        raw_confidences: List[float],
        ground_truths: List[bool]
    ) -> Dict[str, float]:
        """
        Train calibrator on historical data.
        
        Args:
            raw_confidences: List of raw model confidence scores (0-1)
            ground_truths: List of whether claims were actually true
            
        Returns:
            Training metrics
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression
            import numpy as np
            
            X = np.array(raw_confidences).reshape(-1, 1)
            y = np.array(ground_truths, dtype=float)
            
            if self.method == "isotonic":
                self._calibrator = IsotonicRegression(out_of_bounds='clip')
                self._calibrator.fit(raw_confidences, y)
            else:  # platt scaling
                self._calibrator = LogisticRegression()
                self._calibrator.fit(X, y)
            
            # Save model
            os.makedirs(self.output_dir, exist_ok=True)
            model_path = os.path.join(self.output_dir, f"calibrator_{self.method}.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(self._calibrator, f)
            
            logger.info(f"Calibrator saved to {model_path}")
            
            # Compute training metrics
            if self.method == "isotonic":
                calibrated = self._calibrator.predict(raw_confidences)
            else:
                calibrated = self._calibrator.predict_proba(X)[:, 1]
            
            # ECE on training data
            ece = self._compute_ece(calibrated, y)
            brier = np.mean((calibrated - y) ** 2)
            
            return {
                "method": self.method,
                "samples": len(raw_confidences),
                "ece": ece,
                "brier_score": brier,
                "model_path": model_path
            }
            
        except ImportError:
            raise ImportError(
                "scikit-learn required. Install with: pip install scikit-learn"
            )
    
    def _compute_ece(
        self,
        confidences: List[float],
        accuracies: List[float],
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        import numpy as np
        
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > lower) & (confidences <= upper)
            
            if np.sum(in_bin) == 0:
                continue
            
            avg_conf = np.mean(confidences[in_bin])
            avg_acc = np.mean(accuracies[in_bin])
            weight = np.sum(in_bin) / len(confidences)
            
            ece += weight * np.abs(avg_acc - avg_conf)
        
        return float(ece)
    
    def load(self, model_path: str):
        """Load a trained calibrator."""
        with open(model_path, 'rb') as f:
            self._calibrator = pickle.load(f)
        logger.info(f"Loaded calibrator from {model_path}")
    
    def calibrate(self, raw_confidence: float) -> float:
        """Apply calibration to a raw confidence score."""
        if self._calibrator is None:
            raise ValueError("Calibrator not trained. Call train() or load() first.")
        
        import numpy as np
        
        if self.method == "isotonic":
            return float(self._calibrator.predict([raw_confidence])[0])
        else:
            X = np.array([[raw_confidence]])
            return float(self._calibrator.predict_proba(X)[0, 1])


def collect_calibration_data(
    audit_log_path: str
) -> Tuple[List[float], List[bool]]:
    """
    Collect calibration training data from audit logs.
    
    Args:
        audit_log_path: Path to JSON lines file with audit records
        
    Returns:
        raw_confidences, ground_truths
    """
    raw_confidences = []
    ground_truths = []
    
    with open(audit_log_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                confidence = record.get("confidence", 0)
                is_correct = record.get("is_correct", False)
                
                raw_confidences.append(confidence)
                ground_truths.append(is_correct)
            except json.JSONDecodeError:
                continue
    
    return raw_confidences, ground_truths


def evaluate_calibration(
    confidences: List[float],
    accuracies: List[bool],
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Evaluate calibration quality.
    
    Returns:
        ECE, Brier score, reliability diagram data
    """
    import numpy as np
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies, dtype=float)
    
    # ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        count = np.sum(in_bin)
        
        if count == 0:
            bin_data.append({
                "bin": f"{lower:.1f}-{upper:.1f}",
                "count": 0,
                "avg_confidence": (lower + upper) / 2,
                "avg_accuracy": 0
            })
            continue
        
        avg_conf = float(np.mean(confidences[in_bin]))
        avg_acc = float(np.mean(accuracies[in_bin]))
        weight = count / len(confidences)
        
        ece += weight * np.abs(avg_acc - avg_conf)
        
        bin_data.append({
            "bin": f"{lower:.1f}-{upper:.1f}",
            "count": int(count),
            "avg_confidence": avg_conf,
            "avg_accuracy": avg_acc
        })
    
    # Brier score
    brier = float(np.mean((confidences - accuracies) ** 2))
    
    return {
        "ece": float(ece),
        "brier_score": brier,
        "reliability_diagram": bin_data
    }
