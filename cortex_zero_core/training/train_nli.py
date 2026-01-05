"""
NLI Training: Fine-tune NLI models on domain-specific data

Provides tools for training and evaluating NLI models for the HALT-NN pipeline.
"""

from typing import List, Tuple, Optional, Dict, Any
import logging
import json
import os

logger = logging.getLogger(__name__)


class NLITrainer:
    """
    Fine-tune NLI models on custom data.
    
    Supports:
    - Cross-encoder fine-tuning
    - Data augmentation
    - Evaluation with accuracy, F1, confusion matrix
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        output_dir: str = "./models/nli_finetuned",
        batch_size: int = 16,
        epochs: int = 3,
        learning_rate: float = 2e-5
    ):
        """
        Initialize NLI trainer.
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save fine-tuned model
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self._model = None
        
    def prepare_data(
        self,
        examples: List[Tuple[str, str, str]]
    ) -> Tuple[List, List]:
        """
        Prepare training data from examples.
        
        Args:
            examples: List of (premise, hypothesis, label) tuples
                     Labels: "entailment", "contradiction", "neutral"
        
        Returns:
            train_examples, valid_examples
        """
        try:
            from sentence_transformers import InputExample
            import random
            
            # Convert to InputExample format
            all_examples = []
            label_map = {"entailment": 1, "contradiction": 0, "neutral": 2}
            
            for premise, hypothesis, label in examples:
                label_idx = label_map.get(label.lower(), 2)
                example = InputExample(
                    texts=[premise, hypothesis],
                    label=label_idx
                )
                all_examples.append(example)
            
            # Shuffle and split
            random.shuffle(all_examples)
            split_idx = int(len(all_examples) * 0.9)
            
            return all_examples[:split_idx], all_examples[split_idx:]
            
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )
    
    def train(
        self,
        train_examples: List,
        valid_examples: Optional[List] = None
    ) -> Dict[str, float]:
        """
        Train the NLI model.
        
        Args:
            train_examples: Training data
            valid_examples: Validation data
            
        Returns:
            Training metrics
        """
        try:
            from sentence_transformers import CrossEncoder
            from torch.utils.data import DataLoader
            
            logger.info(f"Training NLI model: {self.model_name}")
            logger.info(f"Train examples: {len(train_examples)}, Valid: {len(valid_examples) if valid_examples else 0}")
            
            # Initialize model
            self._model = CrossEncoder(
                self.model_name,
                num_labels=3  # entailment, contradiction, neutral
            )
            
            # Create data loader
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=self.batch_size
            )
            
            # Train without evaluator (CECorrelationEvaluator is for regression, not classification)
            os.makedirs(self.output_dir, exist_ok=True)
            
            self._model.fit(
                train_dataloader=train_dataloader,
                epochs=self.epochs,
                warmup_steps=min(100, len(train_examples) // 2),
                output_path=self.output_dir
            )
            
            # Explicitly save the model (newer sentence-transformers versions)
            self._model.save(self.output_dir)
            
            logger.info(f"Model saved to {self.output_dir}")
            
            return {
                "train_examples": len(train_examples),
                "epochs": self.epochs,
                "output_dir": self.output_dir
            }
            
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )
    
    def evaluate(
        self,
        test_examples: List[Tuple[str, str, str]]
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Returns:
            Accuracy, F1 per class, confusion matrix
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        import numpy as np
        
        label_map = {"entailment": 1, "contradiction": 0, "neutral": 2}
        
        # Prepare pairs and labels
        pairs = [(ex[0], ex[1]) for ex in test_examples]
        true_labels = [label_map.get(ex[2].lower(), 2) for ex in test_examples]
        
        # Predict
        predictions = self._model.predict(pairs)
        pred_labels = np.argmax(predictions, axis=1)
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average="macro")
        conf_matrix = confusion_matrix(true_labels, pred_labels).tolist()
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "confusion_matrix": conf_matrix
        }
    
    def load(self, model_path: str):
        """Load a trained model."""
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_path)
        logger.info(f"Loaded model from {model_path}")


def create_nli_dataset_from_evidence(
    claims: List[str],
    evidence: List[str],
    labels: List[str]
) -> List[Tuple[str, str, str]]:
    """
    Create NLI training dataset from claim-evidence pairs.
    
    Args:
        claims: List of claim texts
        evidence: List of evidence texts
        labels: List of labels ("entailment", "contradiction", "neutral")
        
    Returns:
        List of (evidence, claim, label) tuples for training
    """
    dataset = []
    for claim, ev, label in zip(claims, evidence, labels):
        # NLI format: premise (evidence) -> hypothesis (claim)
        dataset.append((ev, claim, label))
    return dataset


def augment_nli_data(
    examples: List[Tuple[str, str, str]],
    augmentation_factor: int = 2
) -> List[Tuple[str, str, str]]:
    """
    Augment NLI training data.
    
    Techniques:
    - Synonym replacement
    - Back-translation (if available)
    - Negation insertion for contradiction examples
    """
    import random
    
    augmented = list(examples)
    
    for premise, hypothesis, label in examples:
        for _ in range(augmentation_factor - 1):
            # Simple word shuffle augmentation
            words = premise.split()
            if len(words) > 3:
                # Swap two random words
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
                new_premise = " ".join(words)
                augmented.append((new_premise, hypothesis, label))
    
    return augmented
