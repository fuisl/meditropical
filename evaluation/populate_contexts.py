#!/usr/bin/env python3
"""
Script to populate contexts field from RAG JSON files to non-RAG JSON files.
Matches test cases by exact question match.
"""

import json
from pathlib import Path
from typing import Dict, List


def load_json(filepath: Path) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(filepath: Path, data: Dict) -> None:
    """Save JSON file with formatting."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def populate_contexts(rag_file: Path, non_rag_file: Path) -> None:
    """
    Populate contexts from RAG file to non-RAG file by matching questions.
    
    Args:
        rag_file: Path to the RAG JSON file (source)
        non_rag_file: Path to the non-RAG JSON file (target)
    """
    print(f"Processing: {non_rag_file.name}")
    
    # Load both files
    rag_data = load_json(rag_file)
    non_rag_data = load_json(non_rag_file)
    
    # Create a mapping of questions to contexts from RAG file
    question_to_contexts = {}
    for test_case in rag_data.get('test_cases', []):
        question = test_case.get('question')
        contexts = test_case.get('contexts')
        if question:
            question_to_contexts[question] = contexts
    
    # Update non-RAG file with contexts
    updated_count = 0
    for test_case in non_rag_data.get('test_cases', []):
        question = test_case.get('question')
        if question in question_to_contexts:
            test_case['contexts'] = question_to_contexts[question]
            updated_count += 1
    
    # Save updated non-RAG file
    save_json(non_rag_file, non_rag_data)
    print(f"  ✓ Updated {updated_count} test cases")


def main():
    """Main function to process all file pairs in generated_answers directory."""
    generated_answers_dir = Path(__file__).parent / "generated_answers"
    
    if not generated_answers_dir.exists():
        print(f"Error: Directory not found: {generated_answers_dir}")
        return
    
    # Find all RAG files
    rag_files = list(generated_answers_dir.glob("*_rag.json"))
    
    if not rag_files:
        print("No RAG files found in generated_answers directory")
        return
    
    print(f"Found {len(rag_files)} RAG file(s)\n")
    
    processed = 0
    for rag_file in sorted(rag_files):
        # Derive non-RAG filename by removing '_rag' suffix
        non_rag_filename = rag_file.name.replace("_rag.json", ".json")
        non_rag_file = generated_answers_dir / non_rag_filename
        
        if not non_rag_file.exists():
            print(f"Warning: Non-RAG file not found for {rag_file.name}")
            print(f"  Expected: {non_rag_filename}\n")
            continue
        
        try:
            populate_contexts(rag_file, non_rag_file)
            processed += 1
        except Exception as e:
            print(f"  ✗ Error processing {non_rag_file.name}: {e}\n")
    
    print(f"\nCompleted! Processed {processed} file pair(s)")


if __name__ == "__main__":
    main()
