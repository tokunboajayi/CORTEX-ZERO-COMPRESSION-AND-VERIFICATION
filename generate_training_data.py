"""
Generate NLI training data using Gemini API

Creates diverse entailment, contradiction, and neutral examples.
"""

import os
import json
import time

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBFB2FIwI2v6jnAcolk6ic2OQcl-xOnILs"

import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-2.0-flash")

PROMPT = """Generate 20 diverse NLI (Natural Language Inference) training examples.

For each example, provide:
1. A premise (a factual statement)
2. A hypothesis (a claim to evaluate)
3. A label: "entailment" (hypothesis follows from premise), "contradiction" (hypothesis contradicts premise), or "neutral" (cannot be determined)

Cover these topics: science, technology, history, geography, programming, AI, medicine, sports, music, movies.

Output as JSON array with format:
[
  {"premise": "...", "hypothesis": "...", "label": "entailment"},
  {"premise": "...", "hypothesis": "...", "label": "contradiction"},
  ...
]

Make examples diverse and challenging. Include subtle contradictions and neutral cases.
Return ONLY the JSON array, no other text.
"""

def generate_batch():
    """Generate a batch of training examples."""
    try:
        response = model.generate_content(PROMPT)
        text = response.text.strip()
        
        # Extract JSON from response
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        examples = json.loads(text.strip())
        return examples
    except Exception as e:
        print(f"Error: {e}")
        return []

def main():
    output_file = "datasets/combined_nli.jsonl"
    
    # Count existing
    existing = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing = sum(1 for _ in f)
    
    print(f"Existing samples: {existing}")
    print("Generating new samples with Gemini...")
    
    total_new = 0
    batches = 5  # Generate 5 batches of 20 = 100 new samples
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for i in range(batches):
            print(f"\nBatch {i+1}/{batches}...")
            examples = generate_batch()
            
            for ex in examples:
                if all(k in ex for k in ['premise', 'hypothesis', 'label']):
                    f.write(json.dumps(ex) + '\n')
                    total_new += 1
            
            print(f"  Added {len(examples)} examples")
            time.sleep(1)  # Rate limiting
    
    print(f"\n[OK] Generated {total_new} new training samples!")
    print(f"Total dataset size: {existing + total_new} samples")

if __name__ == "__main__":
    main()
