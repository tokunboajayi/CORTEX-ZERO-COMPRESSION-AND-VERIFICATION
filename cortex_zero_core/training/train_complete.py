"""
Complete Training Script for HALT-NN

This script demonstrates how to train both the NLI model and calibrator
using sample data. Run this as a starting point for your own training.

Usage:
    python cortex_zero_core/training/train_complete.py --all
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from cortex_zero_core.training.train_nli import NLITrainer, augment_nli_data
from cortex_zero_core.training.train_calibrator import CalibratorTrainer, evaluate_calibration


# =============================================================================
# SAMPLE TRAINING DATA
# =============================================================================

# NLI Training Data: (premise, hypothesis, label)
SAMPLE_NLI_DATA = [
    # Python facts
    ("Python is a high-level programming language created by Guido van Rossum.",
     "Guido van Rossum created Python.", "entailment"),
    ("Python is a high-level programming language created by Guido van Rossum.",
     "Python was created in the 1990s.", "neutral"),
    ("Python is a high-level programming language created by Guido van Rossum.",
     "Java was created by Guido van Rossum.", "contradiction"),
    
    ("Python uses indentation for code blocks instead of braces.",
     "Python code uses whitespace for structure.", "entailment"),
    ("Python uses indentation for code blocks instead of braces.",
     "Python uses curly braces like C++.", "contradiction"),
    
    ("Python 3.0 was released in 2008.",
     "Python 3 came out before 2010.", "entailment"),
    ("Python 3.0 was released in 2008.",
     "Python 3 was released in 2015.", "contradiction"),
    
    # General facts
    ("The Eiffel Tower is located in Paris, France.",
     "The Eiffel Tower is in France.", "entailment"),
    ("The Eiffel Tower is located in Paris, France.",
     "The Eiffel Tower is in London.", "contradiction"),
    ("The Eiffel Tower is located in Paris, France.",
     "Paris has many tourists.", "neutral"),
    
    ("Water boils at 100 degrees Celsius at sea level.",
     "Water boils at 100°C under normal atmospheric pressure.", "entailment"),
    ("Water boils at 100 degrees Celsius at sea level.",
     "Water freezes at 100 degrees.", "contradiction"),
    
    ("Machine learning is a subset of artificial intelligence.",
     "ML is part of AI.", "entailment"),
    ("Machine learning is a subset of artificial intelligence.",
     "AI is a subset of machine learning.", "contradiction"),
    
    ("The Earth orbits the Sun once every 365.25 days.",
     "A year is approximately 365 days.", "entailment"),
    ("The Earth orbits the Sun once every 365.25 days.",
     "The Sun orbits the Earth.", "contradiction"),
    
    # More entailment examples
    ("JavaScript is the most popular language for web development.",
     "JavaScript is commonly used on websites.", "entailment"),
    ("Neural networks have multiple layers of interconnected nodes.",
     "Neural networks contain layers.", "entailment"),
    ("The Amazon rainforest produces about 20% of the world's oxygen.",
     "The Amazon is important for oxygen production.", "entailment"),
    
    # More contradiction examples
    ("Mars is known as the Red Planet.",
     "Mars is called the Blue Planet.", "contradiction"),
    ("Gravity pulls objects toward the center of mass.",
     "Gravity pushes objects away.", "contradiction"),
    ("DNA contains genetic information.",
     "DNA has no genetic information.", "contradiction"),
    
    # More neutral examples
    ("Tesla was founded in 2003.",
     "Electric cars are becoming popular.", "neutral"),
    ("The Pacific Ocean is the largest ocean.",
     "Whales live in the ocean.", "neutral"),
    ("Coffee contains caffeine.",
     "Many people drink coffee in the morning.", "neutral"),
]

# Calibration Training Data: (raw_confidence, was_correct)
SAMPLE_CALIBRATION_DATA = [
    # High confidence, mostly correct
    (0.95, True), (0.92, True), (0.97, True), (0.88, True), (0.93, False),
    (0.91, True), (0.96, True), (0.89, True), (0.94, True), (0.90, True),
    
    # Medium-high confidence
    (0.82, True), (0.78, True), (0.85, False), (0.75, True), (0.80, True),
    (0.77, False), (0.83, True), (0.79, True), (0.81, True), (0.76, False),
    
    # Medium confidence
    (0.65, True), (0.60, False), (0.68, True), (0.55, False), (0.62, True),
    (0.58, False), (0.67, True), (0.53, False), (0.61, True), (0.56, False),
    
    # Low confidence, mostly wrong
    (0.35, False), (0.28, False), (0.42, True), (0.22, False), (0.38, False),
    (0.31, False), (0.45, True), (0.25, False), (0.33, False), (0.40, True),
    
    # Very low confidence
    (0.15, False), (0.12, False), (0.18, False), (0.08, False), (0.20, True),
]


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def load_dataset_from_file(filepath: str = "./datasets/combined_nli.jsonl"):
    """Load NLI data from JSONL file."""
    import json
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    data.append((
                        entry.get("premise", ""),
                        entry.get("hypothesis", ""),
                        entry.get("label", "neutral")
                    ))
                except json.JSONDecodeError:
                    continue
    return data


def train_nli_model(output_dir: str = "./models/nli_trained"):
    """Train the NLI model on dataset file."""
    print("=" * 60)
    print("TRAINING NLI MODEL")
    print("=" * 60)
    
    # Load data from file
    dataset_file = "./datasets/combined_nli.jsonl"
    if os.path.exists(dataset_file):
        training_data = load_dataset_from_file(dataset_file)
        print(f"\nLoaded {len(training_data)} samples from {dataset_file}")
    else:
        training_data = SAMPLE_NLI_DATA
        print(f"\nUsing {len(training_data)} hardcoded samples (no dataset file)")
    
    # Create trainer
    trainer = NLITrainer(
        model_name="cross-encoder/nli-deberta-v3-base",
        output_dir=output_dir,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )
    
    # Prepare data
    print("\nPreparing data...")
    train_examples, valid_examples = trainer.prepare_data(training_data)
    print(f"Train: {len(train_examples)}, Validation: {len(valid_examples)}")
    
    # Train
    print("\nTraining (this may take a few minutes)...")
    metrics = trainer.train(train_examples, valid_examples)
    
    print(f"\n[OK] Training complete!")
    print(f"   Model saved to: {metrics['output_dir']}")
    print(f"   Epochs: {metrics['epochs']}")
    
    # Evaluate on held-out test data
    test_data = [
        ("Python is widely used in data science.", "Python is popular for ML.", "entailment"),
        ("The moon orbits the Earth.", "The Earth orbits the moon.", "contradiction"),
        ("Coffee is a beverage.", "It rained yesterday.", "neutral"),
    ]
    
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_data)
    print(f"   Accuracy: {results['accuracy']:.2%}")
    print(f"   F1 Macro: {results['f1_macro']:.2%}")
    
    return trainer


def train_calibrator(output_dir: str = "./models/calibrator_trained"):
    """Train the confidence calibrator."""
    print("=" * 60)
    print("TRAINING CALIBRATOR")
    print("=" * 60)
    
    # Extract data
    raw_confidences = [x[0] for x in SAMPLE_CALIBRATION_DATA]
    ground_truths = [x[1] for x in SAMPLE_CALIBRATION_DATA]
    
    print(f"\nTraining samples: {len(raw_confidences)}")
    
    # Pre-training evaluation
    print("\nBefore calibration:")
    pre_eval = evaluate_calibration(raw_confidences, ground_truths)
    print(f"   ECE: {pre_eval['ece']:.4f}")
    print(f"   Brier Score: {pre_eval['brier_score']:.4f}")
    
    # Train
    trainer = CalibratorTrainer(output_dir=output_dir, method="isotonic")
    metrics = trainer.train(raw_confidences, ground_truths)
    
    print(f"\n[OK] Calibrator trained!")
    print(f"   Model saved to: {metrics['model_path']}")
    print(f"   ECE: {metrics['ece']:.4f}")
    print(f"   Brier Score: {metrics['brier_score']:.4f}")
    
    # Demo calibration
    print("\nCalibration examples:")
    test_confs = [0.3, 0.5, 0.7, 0.9]
    for raw in test_confs:
        calibrated = trainer.calibrate(raw)
        print(f"   Raw {raw:.0%} -> Calibrated {calibrated:.0%}")
    
    return trainer


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train HALT-NN models")
    parser.add_argument("--nli", action="store_true", help="Train NLI model")
    parser.add_argument("--calibrator", action="store_true", help="Train calibrator")
    parser.add_argument("--all", action="store_true", help="Train both models")
    parser.add_argument("--output", default="./models", help="Output directory")
    
    args = parser.parse_args()
    
    # Default to --all if no flags
    if not (args.nli or args.calibrator or args.all):
        args.all = True
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.nli or args.all:
        train_nli_model(os.path.join(args.output, "nli_trained"))
        print()
    
    if args.calibrator or args.all:
        train_calibrator(os.path.join(args.output, "calibrator_trained"))
        print()
    
    print("=" * 60)
    print("ALL TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
