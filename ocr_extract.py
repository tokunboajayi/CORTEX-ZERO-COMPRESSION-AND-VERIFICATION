"""
OCR-based Dictionary PDF Extraction

Uses Tesseract OCR and pypdfium2 to extract text from scanned dictionary PDFs.
"""

import os
import re
import json
import random
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
import io
from typing import List, Dict

# Configure Tesseract path (Windows default installation)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# PDF files to process with OCR
PDF_FILES = [
    "largedictionary.pdf",
    "Oxford Collocations dictionary for students of English [www.languagecentre.ir].pdf",
    "OxfordPictureDictionary.pdf"
]

OUTPUT_FILE = "datasets/ocr_dictionary_nli.jsonl"


def ocr_pdf(pdf_path: str, max_pages: int = 20, dpi: int = 200) -> str:
    """Extract text from PDF using OCR."""
    print(f"\nOCR Processing: {os.path.basename(pdf_path)}")
    
    all_text = ""
    
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        pages_to_process = min(len(pdf), max_pages)
        print(f"  Processing {pages_to_process} pages at {dpi} DPI...")
        
        for i in range(pages_to_process):
            # Render page to image
            page = pdf[i]
            bitmap = page.render(scale=dpi/72)  # 72 is default DPI
            pil_image = bitmap.to_pil()
            
            # OCR the image
            text = pytesseract.image_to_string(pil_image, lang='eng')
            all_text += text + "\n"
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{pages_to_process} pages")
        
        pdf.close()
        
    except Exception as e:
        print(f"  Error: {e}")
        return ""
    
    print(f"  Extracted {len(all_text)} characters via OCR")
    return all_text


def extract_dictionary_entries(text: str) -> List[Dict]:
    """Parse OCR text for dictionary entries."""
    entries = []
    
    # Clean up OCR artifacts
    text = re.sub(r'[|\\/_]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Patterns for dictionary entries
    patterns = [
        # word (part of speech) definition
        r'([A-Za-z]{3,20})\s*[\(\[]?(?:n|v|adj|adv|prep|conj|det)\.?[\)\]]?\s*[:—-]?\s*([A-Za-z][^.!?]{10,150}[.!?])',
        # word: definition
        r'([A-Za-z]{3,20})\s*[:\-]\s*([A-Za-z][^.!?]{10,150}[.!?])',
        # CAPITALIZED WORD followed by definition
        r'\n([A-Z]{3,15})\s+([A-Za-z][^.!?]{10,150}[.!?])',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for word, definition in matches:
            word = word.strip().lower()
            definition = definition.strip()
            
            # Filter
            if 3 <= len(word) <= 20 and 15 <= len(definition) <= 300:
                if word.isalpha():  # Only alphabetic words
                    entries.append({
                        "word": word.capitalize(),
                        "definition": definition
                    })
    
    # Remove duplicates
    seen = set()
    unique = []
    for entry in entries:
        key = entry["word"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    
    return unique


def generate_nli_from_definitions(entries: List[Dict]) -> List[Dict]:
    """Convert dictionary entries to NLI examples."""
    examples = []
    words = [e["word"] for e in entries]
    
    for entry in entries:
        word = entry["word"]
        definition = entry["definition"]
        
        # Entailment: definition describes word
        examples.append({
            "premise": f"{word}: {definition}",
            "hypothesis": f"This sentence defines the word {word}.",
            "label": "entailment"
        })
        
        examples.append({
            "premise": definition,
            "hypothesis": f"{word} can be described this way.",
            "label": "entailment"
        })
        
        # Neutral: related but not directly stated
        examples.append({
            "premise": definition,
            "hypothesis": f"{word} is a common English word.",
            "label": "neutral"
        })
        
        # Contradiction: wrong word
        if len(words) > 10:
            other = random.choice([w for w in words if w != word])
            examples.append({
                "premise": f"{word}: {definition}",
                "hypothesis": f"This is the definition of {other}.",
                "label": "contradiction"
            })
    
    return examples


def main():
    print("=" * 60)
    print("OCR DICTIONARY EXTRACTION")
    print("=" * 60)
    
    # Check Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
    except Exception as e:
        print(f"ERROR: Tesseract not found. Please restart your terminal.")
        print(f"  Error: {e}")
        return
    
    all_entries = []
    
    for pdf_file in PDF_FILES:
        if os.path.exists(pdf_file):
            # OCR extract
            text = ocr_pdf(pdf_file, max_pages=15)  # Limit to save time
            
            if text:
                entries = extract_dictionary_entries(text)
                print(f"  Found {len(entries)} dictionary entries")
                all_entries.extend(entries)
        else:
            print(f"File not found: {pdf_file}")
    
    if not all_entries:
        print("\nNo entries extracted. OCR may need adjustment.")
        return
    
    print(f"\nTotal entries: {len(all_entries)}")
    
    # Generate NLI examples
    print("\nGenerating NLI examples...")
    nli_examples = generate_nli_from_definitions(all_entries)
    random.shuffle(nli_examples)
    
    # Save
    os.makedirs("datasets", exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ex in nli_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"\n[OK] Generated {len(nli_examples)} NLI examples")
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Merge with main dataset
    main_dataset = "datasets/combined_nli.jsonl"
    if os.path.exists(main_dataset):
        with open(main_dataset, 'a', encoding='utf-8') as f:
            for ex in nli_examples:
                f.write(json.dumps(ex) + '\n')
        
        with open(main_dataset, 'r') as f:
            total = sum(1 for _ in f)
        print(f"\n[OK] Total dataset size: {total} samples")


if __name__ == "__main__":
    main()
