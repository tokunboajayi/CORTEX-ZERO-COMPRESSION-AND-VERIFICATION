"""
Extract Dictionary PDFs and Convert to NLI Training Data

This script extracts definitions from dictionary PDFs and creates
NLI training examples in the format:
- premise: The definition
- hypothesis: A claim about the word
- label: entailment, contradiction, or neutral
"""

import os
import re
import json
import random
import pdfplumber
from typing import List, Dict, Tuple

# PDF files to process
PDF_FILES = [
    "largedictionary.pdf",
    "Oxford Collocations dictionary for students of English [www.languagecentre.ir].pdf",
    "OxfordPictureDictionary.pdf",
    "The_Oxford_3000.pdf"
]

OUTPUT_FILE = "datasets/dictionary_nli.jsonl"


def extract_text_from_pdf(pdf_path: str, max_pages: int = 50) -> str:
    """Extract text from a PDF file."""
    print(f"Extracting from: {os.path.basename(pdf_path)}")
    
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_read = min(len(pdf.pages), max_pages)
            print(f"  Reading {pages_to_read} pages...")
            
            for i, page in enumerate(pdf.pages[:pages_to_read]):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{pages_to_read} pages")
                    
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"  Extracted {len(text)} characters")
    return text


def parse_dictionary_entries(text: str) -> List[Dict]:
    """Parse dictionary text into word-definition pairs."""
    entries = []
    
    # Pattern for dictionary entries: word followed by definition
    # This handles common dictionary formats
    patterns = [
        # Pattern: word (pos) definition
        r'([A-Za-z]+)\s*\([a-z]+\.\)\s*([^.]+\.)',
        # Pattern: word: definition
        r'([A-Za-z]+):\s*([^.]+\.)',
        # Pattern: word - definition
        r'([A-Za-z]+)\s*[-–]\s*([^.]+\.)',
        # Pattern: bold word followed by text
        r'\n([A-Z][a-z]+)\s+([A-Za-z][^.]+\.)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for word, definition in matches:
            word = word.strip()
            definition = definition.strip()
            
            # Filter valid entries
            if 3 <= len(word) <= 20 and 10 <= len(definition) <= 500:
                entries.append({
                    "word": word,
                    "definition": definition
                })
    
    # Remove duplicates
    seen = set()
    unique_entries = []
    for entry in entries:
        key = (entry["word"].lower(), entry["definition"][:50])
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)
    
    return unique_entries


def generate_nli_examples(entries: List[Dict]) -> List[Dict]:
    """Convert dictionary entries to NLI training examples."""
    nli_examples = []
    
    for entry in entries:
        word = entry["word"]
        definition = entry["definition"]
        
        # 1. ENTAILMENT: Definition implies word is X
        nli_examples.append({
            "premise": definition,
            "hypothesis": f"{word} is described by this definition.",
            "label": "entailment"
        })
        
        # 2. ENTAILMENT: Paraphrased
        nli_examples.append({
            "premise": f"{word}: {definition}",
            "hypothesis": f"This is the definition of {word}.",
            "label": "entailment"
        })
        
        # 3. NEUTRAL: Related but not directly implied
        nli_examples.append({
            "premise": definition,
            "hypothesis": f"{word} is commonly used in everyday language.",
            "label": "neutral"
        })
        
        # 4. CONTRADICTION: Wrong word
        if len(entries) > 1:
            other_word = random.choice([e["word"] for e in entries if e["word"] != word])
            nli_examples.append({
                "premise": definition,
                "hypothesis": f"This is the definition of {other_word}.",
                "label": "contradiction"
            })
    
    return nli_examples


def process_oxford_3000(text: str) -> List[Dict]:
    """Special processing for Oxford 3000 word list."""
    entries = []
    
    # Extract word list with frequency markers
    # Pattern for Oxford 3000: word followed by part of speech
    words = re.findall(r'\b([a-z]+)\s+(?:n\.|v\.|adj\.|adv\.)', text.lower())
    
    for word in set(words):
        if 3 <= len(word) <= 15:
            entries.append({
                "word": word,
                "definition": f"{word.capitalize()} is one of the 3000 most important English words to learn."
            })
    
    return entries


def main():
    print("=" * 60)
    print("DICTIONARY PDF EXTRACTION FOR NLI TRAINING")
    print("=" * 60)
    
    all_entries = []
    
    for pdf_file in PDF_FILES:
        if os.path.exists(pdf_file):
            text = extract_text_from_pdf(pdf_file, max_pages=30)
            
            if "Oxford_3000" in pdf_file:
                entries = process_oxford_3000(text)
            else:
                entries = parse_dictionary_entries(text)
            
            print(f"  Found {len(entries)} dictionary entries")
            all_entries.extend(entries)
        else:
            print(f"File not found: {pdf_file}")
    
    print(f"\nTotal dictionary entries: {len(all_entries)}")
    
    # Generate NLI examples
    print("\nGenerating NLI training examples...")
    nli_examples = generate_nli_examples(all_entries)
    
    # Shuffle for variety
    random.shuffle(nli_examples)
    
    # Save to file
    os.makedirs("datasets", exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for example in nli_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n[OK] Generated {len(nli_examples)} NLI training examples")
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Show distribution
    labels = [ex["label"] for ex in nli_examples]
    print(f"\nLabel distribution:")
    for label in ["entailment", "contradiction", "neutral"]:
        count = labels.count(label)
        pct = count / len(labels) * 100 if labels else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Merge with main dataset
    main_dataset = "datasets/combined_nli.jsonl"
    if os.path.exists(main_dataset):
        with open(main_dataset, 'a', encoding='utf-8') as f:
            for example in nli_examples:
                f.write(json.dumps(example) + '\n')
        print(f"\n[OK] Merged into {main_dataset}")
        
        # Count total
        with open(main_dataset, 'r') as f:
            total = sum(1 for _ in f)
        print(f"Total dataset size: {total} samples")


if __name__ == "__main__":
    main()
