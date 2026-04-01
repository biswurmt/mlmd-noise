#!/usr/bin/env python3
"""
sample_discoveries.py — Extract a random sample of "False Positives" from the 
semantic Knowledge Graph retrieval for manual clinical review.

This script isolates cases where the semantic KG recommended a test (e.g., "Arm X-Ray"), 
but the legacy regex ground truth marked it as 0/False. These are our potential 
"Discoveries" (valid clinical variations missed by the old rules).

Usage:
    python sample_discoveries.py --input-csv patient_diagnoses_semantic.csv \
                                 --test-name "Arm X-Ray" \
                                 --gt-column xray_arm_dx \
                                 --sample-size 100
"""

import argparse
import pandas as pd
import sys

def extract_discovery_sample(
    input_csv: str, 
    output_csv: str, 
    test_name: str, 
    gt_column: str, 
    sample_size: int,
    random_seed: int = 42
):
    print(f"Loading data from '{input_csv}'...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_csv}'.")
        sys.exit(1)

    # Ensure required columns exist
    required_cols = ['dx', 'potential_tests', gt_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {missing_cols}")
        sys.exit(1)

    # List of columns you want to remove
    columns_to_drop = [
        "patient_visit_id_hash", "patient_id_hash", "age", "sex", "weight", 
        "language", "CTAS", "arr_method", "care_area", "patient_CC", 
        "nurse_CC", "pulse", "resp", "temp", "LWBS", "txt", "DistSK", 
        "year", "month", "day", "hour", "systolic", "diastolic", "ecg_di", 
        "xray_arm_di", "us_app_di", "us_testes_di", "order_id", "order_time", "name"
    ]

    # Assuming your data is already in a DataFrame named 'df'
    df = df.drop(columns=columns_to_drop)

    # 1. Identify rows where the Semantic KG predicted the test
    # (Handling NaNs safely by filling with empty string)
    predicted_mask = df['potential_tests'].fillna('').str.contains(test_name, case=False, regex=False)

    # 2. Identify rows where the legacy Ground Truth is Negative (0 or False)
    # (Handling potential boolean or integer representations of the ground truth)
    actual_mask = df[gt_column].isin([0, '0', False, 'False', 0.0])

    # 3. Combine to find the "False Positives" (Discoveries)
    discoveries_df = df[predicted_mask & actual_mask]
    total_discoveries = len(discoveries_df)

    print(f"\nFound {total_discoveries} potential discoveries (KG predicted '{test_name}', but GT '{gt_column}' is negative).")

    if total_discoveries == 0:
        print("No discoveries found. Exiting.")
        sys.exit(0)

    # 4. Take a random sample
    n_sample = min(sample_size, total_discoveries)
    sample_df = discoveries_df.sample(n=n_sample, random_state=random_seed)

    print(f"Sampled {n_sample} records for manual review.")

    # 5. Format the output for easy manual review
    # Keep the essential columns first, followed by everything else
    review_cols = ['dx', 'matched_graph_nodes', 'potential_tests', gt_column]
    other_cols = [c for c in sample_df.columns if c not in review_cols]
    
    # Add a blank column for the human reviewer to mark True/False
    sample_df.insert(0, 'reviewer_is_valid_discovery', '')
    
    final_df = sample_df[['reviewer_is_valid_discovery'] + review_cols + other_cols]

    # Save to CSV
    final_df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Saved review sample to '{output_csv}'")
    print("Please have a clinical domain expert review the 'dx' column and fill out 'reviewer_is_valid_discovery'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a random sample of KG 'False Positives' for manual review."
    )
    parser.add_argument(
        "--input-csv", required=True,
        help="Path to the enriched CSV output from the semantic mapping."
    )
    parser.add_argument(
        "--output-csv", default="discovery_review_sample.csv",
        help="Path to save the sample for manual review (default: discovery_review_sample.csv)."
    )
    parser.add_argument(
        "--test-name", required=True,
        help="The specific test name to evaluate (e.g., 'Arm X-Ray')."
    )
    parser.add_argument(
        "--gt-column", required=True,
        help="The legacy ground truth column corresponding to the test (e.g., 'xray_arm_dx')."
    )
    parser.add_argument(
        "--sample-size", type=int, default=100,
        help="Number of records to sample for review (default: 100)."
    )

    args = parser.parse_args()

    extract_discovery_sample(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        test_name=args.test_name,
        gt_column=args.gt_column,
        sample_size=args.sample_size
    )