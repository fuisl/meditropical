#!/usr/bin/env python3
"""
Script to combine evaluation metrics from multiple JSON files into a CSV.
Extracts metrics from eval_answer, eval_reasoning, and eval_answer_relevancy files
for different model configurations.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any


def parse_filename(filename: str) -> tuple:
    """
    Parse filename to extract evaluation type and model configuration.
    
    Returns:
        tuple: (eval_type, model_name, use_rag)
    """
    # Remove .json extension
    name = filename.replace('.json', '')
    
    # Determine evaluation type
    if name.startswith('eval_answer_relevancy_'):
        eval_type = 'answer_relevancy'
        model_part = name.replace('eval_answer_relevancy_', '')
    elif name.startswith('eval_reasoning_'):
        eval_type = 'reasoning'
        model_part = name.replace('eval_reasoning_', '')
    elif name.startswith('eval_answer_'):
        eval_type = 'answer'
        model_part = name.replace('eval_answer_', '')
    else:
        return None, None, None
    
    # Check if RAG is used
    if model_part.endswith('_rag'):
        use_rag = True
        model_name = model_part.replace('_generated_cases_rag', '')
    else:
        use_rag = False
        model_name = model_part.replace('_generated_cases', '')
    
    return eval_type, model_name, use_rag


def get_model_display_name(model_name: str, use_rag: bool) -> str:
    """Get a clean display name for the model configuration."""
    # Map technical names to display names
    name_map = {
        'google_gemma-3-4b-it': 'Gemma 3 4B',
        'google_gemma-3-4b-it-finetuned': 'Gemma 3 4B Finetuned',
        'google_medgemma-4b-it': 'MedGemma 4B'
    }
    
    base_name = name_map.get(model_name, model_name)
    
    if use_rag:
        return f"{base_name} + RAG"
    return base_name


def extract_metrics(json_file: Path) -> Dict[str, Any]:
    """Extract metrics from a JSON evaluation file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        benchmark_stats = data.get('benchmark_stats', {})
        average_metrics = benchmark_stats.get('average_metrics', {})
        
        return {
            'total_tests': benchmark_stats.get('total_tests', 0),
            'success_rate': benchmark_stats.get('success_rate', 0.0),
            'metrics': average_metrics,
            'elapsed_time': data.get('elapsed_time_seconds', 0.0)
        }
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return None


def main():
    """Main function to process all evaluation files and generate CSV."""
    results_dir = Path(__file__).parent / 'results'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Collect all evaluation files
    eval_files = list(results_dir.glob('eval_*.json'))
    
    print(f"Found {len(eval_files)} evaluation files")
    
    # Organize data by model configuration
    model_configs = {}
    
    for eval_file in eval_files:
        eval_type, model_name, use_rag = parse_filename(eval_file.name)
        
        if eval_type is None:
            continue
        
        # Create unique key for this configuration
        config_key = (model_name, use_rag)
        
        if config_key not in model_configs:
            model_configs[config_key] = {
                'model_name': model_name,
                'use_rag': use_rag,
                'display_name': get_model_display_name(model_name, use_rag),
                'answer': {},
                'reasoning': {},
                'answer_relevancy': {}
            }
        
        # Extract metrics
        metrics_data = extract_metrics(eval_file)
        if metrics_data:
            model_configs[config_key][eval_type] = metrics_data
    
    # Prepare CSV data
    csv_rows = []
    
    # Define the order of configurations
    config_order = [
        ('google_medgemma-4b-it', False),
        ('google_medgemma-4b-it', True),
        ('google_gemma-3-4b-it', False),
        ('google_gemma-3-4b-it', True),
        ('google_gemma-3-4b-it-finetuned', False),
        ('google_gemma-3-4b-it-finetuned', True),
    ]
    
    for config_key in config_order:
        if config_key not in model_configs:
            continue
        
        config = model_configs[config_key]
        
        row = {
            'Model': config['display_name'],
            'RAG': 'Yes' if config['use_rag'] else 'No',
        }
        
        # Extract answer metrics
        if 'answer' in config and config['answer']:
            answer_metrics = config['answer'].get('metrics', {})
            row['Faithfulness'] = f"{answer_metrics.get('faithfulness', 0):.4f}"
            # row['Answer Relevance (Answer)'] = f"{answer_metrics.get('answer_relevance', 0):.4f}"
            row['Answer Accuracy'] = f"{answer_metrics.get('answer_accuracy', 0):.4f}"
            # row['RAGAS Score (Answer)'] = f"{answer_metrics.get('ragas_score', 0):.4f}"
        
        # Extract reasoning metrics
        if 'reasoning' in config and config['reasoning']:
            reasoning_metrics = config['reasoning'].get('metrics', {})
            row['Reasoning Recall'] = f"{reasoning_metrics.get('reasoning_recall', 0):.4f}"
            # row['RAGAS Score (Reasoning)'] = f"{reasoning_metrics.get('ragas_score', 0):.4f}"
        
        # Extract answer relevancy metrics
        if 'answer_relevancy' in config and config['answer_relevancy']:
            relevancy_metrics = config['answer_relevancy'].get('metrics', {})
            row['Answer Relevance'] = f"{relevancy_metrics.get('answer_relevance', 0):.4f}"
            # row['RAGAS Score (Relevancy)'] = f"{relevancy_metrics.get('ragas_score', 0):.4f}"
        
        # Add test counts and success rate
        for eval_type in ['answer', 'reasoning', 'answer_relevancy']:
            if eval_type in config and config[eval_type]:
                row['Total Tests'] = config[eval_type].get('total_tests', 0)
                row['Success Rate (%)'] = f"{config[eval_type].get('success_rate', 0):.1f}"
                break
        
        csv_rows.append(row)
    
    # Write CSV file
    output_file = results_dir / 'combined_metrics.csv'
    
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"\n✓ Combined metrics saved to: {output_file}")
        print(f"  Total configurations: {len(csv_rows)}")
        
        # Print summary table
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        for row in csv_rows:
            print(f"\n{row['Model']}")
            for key, value in row.items():
                if key not in ['Model', 'RAG']:
                    print(f"  {key}: {value}")
    else:
        print("No data to write to CSV")


if __name__ == '__main__':
    main()
