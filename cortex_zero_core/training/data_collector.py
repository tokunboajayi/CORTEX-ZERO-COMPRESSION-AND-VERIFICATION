"""
Data Collector: Automatic collection of training data from API usage

Logs all queries and verification results to build a training dataset
from real-world usage patterns.
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects training data from HALT-NN API usage.
    
    Logs:
    - Queries and their verification results
    - Claim-evidence pairs with NLI predictions
    - Confidence scores (for calibration training)
    - User feedback (correct/incorrect)
    """
    
    def __init__(
        self,
        log_dir: str = "./training_data",
        auto_save_interval: int = 100
    ):
        """
        Initialize data collector.
        
        Args:
            log_dir: Directory to save training data
            auto_save_interval: Auto-save after this many entries
        """
        self.log_dir = log_dir
        self.auto_save_interval = auto_save_interval
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Data buffers
        self.nli_data: List[Dict] = []
        self.calibration_data: List[Dict] = []
        self.feedback_data: List[Dict] = []
        
        # File paths
        self.nli_file = os.path.join(log_dir, "nli_data.jsonl")
        self.calibration_file = os.path.join(log_dir, "calibration_data.jsonl")
        self.feedback_file = os.path.join(log_dir, "feedback_data.jsonl")
        
        self._entry_count = 0
    
    def log_verification(
        self,
        query: str,
        claims: List[Dict],
        evidence: List[Dict],
        links: List[Dict],
        confidence: float,
        action: str
    ):
        """
        Log a complete verification result.
        
        Args:
            query: Original query
            claims: List of claims with status
            evidence: List of evidence used
            links: NLI links between claims and evidence
            confidence: Overall confidence score
            action: Final action (ANSWER, ABSTAIN, etc.)
        """
        timestamp = datetime.now().isoformat()
        
        # Log NLI data (claim-evidence pairs)
        for link in links:
            claim = next((c for c in claims if c.get('id') == link.get('claim_id')), {})
            ev = next((e for e in evidence if e.get('id') == link.get('evidence_id')), {})
            
            if claim and ev:
                nli_entry = {
                    "timestamp": timestamp,
                    "premise": ev.get("content", ""),
                    "hypothesis": claim.get("text", ""),
                    "predicted_label": link.get("nli_label", "NEUTRAL"),
                    "probability": link.get("nli_probability", 0.0),
                    "claim_status": claim.get("status", ""),
                    "source_tier": ev.get("tier", "B")
                }
                self.nli_data.append(nli_entry)
                self._append_jsonl(self.nli_file, nli_entry)
        
        # Log calibration data
        for claim in claims:
            cal_entry = {
                "timestamp": timestamp,
                "claim_text": claim.get("text", ""),
                "confidence": claim.get("confidence", 0.0),
                "status": claim.get("status", ""),
                "evidence_count": len(claim.get("evidence_ids", [])),
                "overall_confidence": confidence,
                "action": action
            }
            self.calibration_data.append(cal_entry)
            self._append_jsonl(self.calibration_file, cal_entry)
        
        self._entry_count += 1
        
        if self._entry_count % self.auto_save_interval == 0:
            logger.info(f"Data collector: {self._entry_count} entries logged")
    
    def log_feedback(
        self,
        query: str,
        claim_text: str,
        was_correct: bool,
        user_correction: Optional[str] = None
    ):
        """
        Log user feedback on verification results.
        
        Args:
            query: Original query
            claim_text: The claim being evaluated
            was_correct: Whether the verification was correct
            user_correction: Optional correct answer
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "claim_text": claim_text,
            "was_correct": was_correct,
            "user_correction": user_correction
        }
        self.feedback_data.append(entry)
        self._append_jsonl(self.feedback_file, entry)
    
    def _append_jsonl(self, filepath: str, entry: Dict):
        """Append entry to JSONL file."""
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    
    def export_nli_training_data(self) -> List[tuple]:
        """
        Export NLI data in training format.
        
        Returns:
            List of (premise, hypothesis, label) tuples
        """
        data = []
        
        # Load from file
        if os.path.exists(self.nli_file):
            with open(self.nli_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        # Map NLI labels to training format
                        label_map = {
                            "ENTAILS": "entailment",
                            "CONTRADICTS": "contradiction",
                            "NEUTRAL": "neutral"
                        }
                        label = label_map.get(entry.get("predicted_label", ""), "neutral")
                        
                        data.append((
                            entry.get("premise", ""),
                            entry.get("hypothesis", ""),
                            label
                        ))
                    except json.JSONDecodeError:
                        continue
        
        return data
    
    def export_calibration_data(self) -> tuple:
        """
        Export calibration data for training.
        
        Returns:
            (confidences, ground_truths) where ground_truths come from feedback
        """
        confidences = []
        ground_truths = []
        
        # Load feedback to get ground truth
        feedback_map = {}
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        key = entry.get("claim_text", "")
                        feedback_map[key] = entry.get("was_correct", False)
                    except json.JSONDecodeError:
                        continue
        
        # Match calibration data with feedback
        if os.path.exists(self.calibration_file):
            with open(self.calibration_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        claim_text = entry.get("claim_text", "")
                        
                        if claim_text in feedback_map:
                            confidences.append(entry.get("confidence", 0.0))
                            ground_truths.append(feedback_map[claim_text])
                    except json.JSONDecodeError:
                        continue
        
        return confidences, ground_truths
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        nli_count = 0
        cal_count = 0
        feedback_count = 0
        
        if os.path.exists(self.nli_file):
            with open(self.nli_file, 'r') as f:
                nli_count = sum(1 for _ in f)
        
        if os.path.exists(self.calibration_file):
            with open(self.calibration_file, 'r') as f:
                cal_count = sum(1 for _ in f)
        
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                feedback_count = sum(1 for _ in f)
        
        return {
            "nli_entries": nli_count,
            "calibration_entries": cal_count,
            "feedback_entries": feedback_count,
            "log_dir": self.log_dir
        }


# Global collector instance
_collector: Optional[DataCollector] = None


def get_collector(log_dir: str = "./training_data") -> DataCollector:
    """Get or create the global data collector."""
    global _collector
    if _collector is None:
        _collector = DataCollector(log_dir=log_dir)
    return _collector
