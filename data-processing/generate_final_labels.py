import pandas as pd
import json

# 1. Load the CSV with the matches_json column
csv_path = 'patient_diagnoses_semantic_chroma_enriched_v3_filtered.csv'
print(f"Loading CSV: {csv_path}...")
df = pd.read_csv(csv_path)

# Ensure the 'dx' column is unique
initial_len = len(df)
df = df.drop_duplicates(subset=['dx'], keep='first').reset_index(drop=True)
print(f"Dropped {initial_len - len(df)} duplicate diagnoses. Working with {len(df)} unique rows.")

# ==========================================
# 2. YOUR GOLDEN CONFIGURATION
# ==========================================
_DEFAULT_SEMANTIC_THRESHOLD = 0.70  
_TEST_THRESHOLDS = {
    "ECG":                   0.70,
    "Appendix Ultrasound":   0.69,
    "Arm X-Ray":             0.74,
    "Testicular Ultrasound": 0.74,
}

# REPLACED MIN_NODES WITH AGGREGATE SCORES
_DEFAULT_MIN_SCORE_SUM = 0.0  # If it only needs 1 node, any passing node is enough
_TEST_MIN_SCORE_SUM = {
    "Appendix Ultrasound":   1.38,  # Consensus: Requires roughly 2 valid nodes (2 x 0.69)
    "Testicular Ultrasound": 1.48,  # Consensus: Requires roughly 2 valid nodes (2 x 0.74)
}

# ==========================================
# 3. THE DYNAMIC SCORE AGGREGATION LOGIC
# ==========================================
def apply_dynamic_score_rules_to_json(matches_json_str):
    if pd.isna(matches_json_str) or not str(matches_json_str).strip():
        return "No linked tests found"

    try:
        matches = json.loads(matches_json_str)
    except Exception:
        return "No linked tests found"

    test_score_sum = {}
    
    # Step A: Aggregate the cosine scores of valid nodes
    for m in matches:
        node_score = m.get('score', 0.0)
        
        for t in m.get('tests', []):
            # Get the specific threshold for this test, or use the default
            req_threshold = _TEST_THRESHOLDS.get(t, _DEFAULT_SEMANTIC_THRESHOLD)
            
            # If the node's score is high enough for THIS test, add its score to the aggregate!
            if node_score >= req_threshold:
                test_score_sum[t] = test_score_sum.get(t, 0.0) + node_score

    # Step B: Filter by test-specific minimum aggregate score (Consensus)
    valid_tests = []
    for t, total_score in test_score_sum.items():
        req_min_score = _TEST_MIN_SCORE_SUM.get(t, _DEFAULT_MIN_SCORE_SUM)
        
        if total_score >= req_min_score:
            valid_tests.append(t)
            
    return ", ".join(valid_tests) if valid_tests else "No linked tests found"

# ==========================================
# 4. GENERATE NEW LABELS
# ==========================================
print("Applying score-weighted dynamic rules to generate final predictions...")
df['final_potential_tests'] = df['matches_json'].apply(apply_dynamic_score_rules_to_json)

# Create the clean binary _kg columns based on the final text
print("Building binary _kg prediction columns...")
df['ecg_kg'] = df['final_potential_tests'].apply(lambda x: 1 if 'ECG' in str(x) else 0)
df['xray_arm_kg'] = df['final_potential_tests'].apply(lambda x: 1 if 'Arm X-Ray' in str(x) else 0)
df['us_app_kg'] = df['final_potential_tests'].apply(lambda x: 1 if 'Appendix Ultrasound' in str(x) else 0)
df['us_testes_kg'] = df['final_potential_tests'].apply(lambda x: 1 if 'Testicular Ultrasound' in str(x) else 0)

# Save the finalized dataset
output_path = 'patient_diagnoses_FINAL_PREDICTIONS.csv'
df.to_csv(output_path, index=False)
print(f"\nSuccess! Final dataset saved to {output_path}")

# ==========================================
# 5. THE FINAL VALIDATION REPORT
# ==========================================
test_mapping = {
    'ecg_dx': ('ecg_kg', 'ECG'),
    'xray_arm_dx': ('xray_arm_kg', 'Arm X-Ray'),
    'us_app_dx': ('us_app_kg', 'Appendix Ultrasound'),
    'us_testes_dx': ('us_testes_kg', 'Testicular Ultrasound')
}

print("\n" + "="*95)
print("FINAL VALIDATION REPORT (SCORE-WEIGHTED DYNAMIC CONFIGURATION)")
print("="*95)
print(f"{'Test':<25} {'GT Column':<15} {'TP':>5} {'FP':>6} {'TN':>7} {'FN':>5} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}")
print("-" * 95)

validation_rows = []
for gt_col, (pred_col, test_name) in test_mapping.items():
    actual_series = df[gt_col]
    predicted_series = df[pred_col]

    tp = ((actual_series == 1) & (predicted_series == 1)).sum()
    tn = ((actual_series == 0) & (predicted_series == 0)).sum()
    fp = ((actual_series == 0) & (predicted_series == 1)).sum()
    fn = ((actual_series == 1) & (predicted_series == 0)).sum()

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print(f"{test_name:<25} {gt_col:<15} {tp:>5} {fp:>6} {tn:>7} {fn:>5} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {acc:>6.3f}")
    validation_rows.append({
        "test_name": test_name,
        "gt_column": gt_col,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(acc, 4),
    })
print("-" * 95)

validation_path = output_path.replace(".csv", "_validation.csv")
pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
print(f"\nValidation report saved to {validation_path}")