import pandas as pd
from pathlib import Path
import numpy as np

data_dir = Path('data/fire/')
pred_dir = Path('llama_predictions/')

# test file
test_file = 'nq_test.jsonl'

# prediction file
pred_standard_file = 'nq_test.jsonl_standard_generation.output_topp0.0_genlen32.jsonl'
pred_adacad_file = 'nq_test.jsonl_adacad.output_topp0.0_genlen32.jsonl'
assert pred_standard_file.startswith(test_file), f"Prediction file should be based on test file: {test_file}"
assert pred_adacad_file.startswith(test_file), f"Prediction file should be based on test file: {test_file}"


test_filepath = data_dir / test_file
pred_standard_filepath = data_dir / pred_dir / pred_standard_file
pred_adacad_filepath = data_dir / pred_dir / pred_adacad_file


def read_json(filepath):
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_json(filepath, lines=True)

# get labels for each instance. Note: each instance is duplicated context+question, and question, so use odd/even idices only
test_df = read_json(filepath=test_filepath)
odd_indices = range(1, len(test_df), 2)
odd_gold_answers = test_df.iloc[odd_indices]['gold_answers']
gold_labels = odd_gold_answers
unique_labels = set(gold_labels.tolist())

# gold predictions for context-query standard generation
pred_standard_df = read_json(filepath=pred_standard_filepath)
even_indices = range(0, len(pred_standard_df), 2)
context_query_preds = pred_standard_df.iloc[even_indices]['string'].apply(lambda x: x[0])

# gold predictions for query standard generation
pred_standard_df = read_json(filepath=pred_standard_filepath)
odd_indices = range(1, len(pred_standard_df), 2)
query_preds = pred_standard_df.iloc[odd_indices]['string'].apply(lambda x: x[0])

# adacad predictions 
pred_adacad_df = read_json(filepath=pred_adacad_filepath)
adacad_preds = pred_adacad_df['string'].apply(lambda x: x[0])

def clean_prediction(text, valid_labels):
    """Find first valid label that appears in the text"""
    text_lower = str(text).lower()
    for label in valid_labels:
        if str(label).lower() in text_lower:
            return label
    return ""

# Clean predictions
context_query_preds = context_query_preds.apply(lambda x: clean_prediction(x, unique_labels)).tolist()
query_preds = query_preds.apply(lambda x: clean_prediction(x, unique_labels)).tolist()
adacad_preds = adacad_preds.apply(lambda x: clean_prediction(x, unique_labels)).tolist()

cq_empty_count = context_query_preds.count("")
q_empty_count = query_preds.count("")
a_empty_count = adacad_preds.count("")

print(f"Empty predictions:")
print(f"  Context+Query: {cq_empty_count}/{len(context_query_preds)} ({cq_empty_count/len(context_query_preds)*100:.1f}%)")
print(f"  Query only:    {q_empty_count}/{len(query_preds)} ({q_empty_count/len(query_preds)*100:.1f}%)")
print(f"  AdaCAD:        {a_empty_count}/{len(adacad_preds)} ({a_empty_count/len(adacad_preds)*100:.1f}%)")

# Find all unique indices with empty strings in any of the prediction lists
empty_indices = set()

# Find indices of empty strings in each list
for i, pred in enumerate(context_query_preds):
    if pred == "":
        empty_indices.add(i)

for i, pred in enumerate(query_preds):
    if pred == "":
        empty_indices.add(i)
        
for i, pred in enumerate(adacad_preds):
    if pred == "":
        empty_indices.add(i)

# Create filtered lists by keeping only indices NOT in empty_indices
context_query_preds = [pred for i, pred in enumerate(context_query_preds) if i not in empty_indices]
query_preds = [pred for i, pred in enumerate(query_preds) if i not in empty_indices]
adacad_preds = [pred for i, pred in enumerate(adacad_preds) if i not in empty_indices]
gold_labels = [gold for i, gold in enumerate(gold_labels) if i not in empty_indices]

print(f"\nAfter filtering:")
print(f"  Context+Query: {len(context_query_preds)} predictions")
print(f"  Query only:    {len(query_preds)} predictions")
print(f"  AdaCAD:        {len(adacad_preds)} predictions")
print(f"  Removed {len(empty_indices)} instances where the LLM failed to find valid labels")

def get_f1(key, prediction):
    correct_by_relation = ((key == prediction) & (prediction != 0)).astype(np.int32).sum()
    guessed_by_relation = (prediction != 0).astype(np.int32).sum()
    gold_by_relation = (key != 0).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro

# Create label to ID mapping with 'no_relation' as 0
unique_labels_list = list(unique_labels)
if 'no_relation' in unique_labels_list:
    unique_labels_list.remove('no_relation')

# Create mapping: 'no_relation' = 0, other labels = 1, 2, 3, ...
label_to_id = {'no_relation': 0}
for i, label in enumerate(sorted(unique_labels_list), 1):
    label_to_id[label] = i

# Convert string labels to IDs
def map_labels_to_ids(labels, label_mapping):
    """Convert list of string labels to list of IDs"""
    return [label_mapping.get(label, 0) for label in labels]  # default to 0 (no_relation) if not found

# Convert to IDs
gold_labels_ids = map_labels_to_ids(gold_labels, label_to_id)
context_query_ids = map_labels_to_ids(context_query_preds, label_to_id)
query_ids = map_labels_to_ids(query_preds, label_to_id)
adacad_ids = map_labels_to_ids(adacad_preds, label_to_id)

# Convert to numpy arrays for the F1 function
gold_labels_ids = np.array(gold_labels_ids)
context_query_ids = np.array(context_query_ids)
query_ids = np.array(query_ids)
adacad_ids = np.array(adacad_ids)

# Calculate F1 scores
print(f"\nF1 Score Results:")
print(f"{'Method':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 50)

q_prec, q_recall, q_f1 = get_f1(gold_labels_ids, query_ids)
print(f"{'Query only':<15} {q_prec:<10.4f} {q_recall:<10.4f} {q_f1:<10.4f}")

cq_prec, cq_recall, cq_f1 = get_f1(gold_labels_ids, context_query_ids)
print(f"{'Context+Query':<15} {cq_prec:<10.4f} {cq_recall:<10.4f} {cq_f1:<10.4f}")

a_prec, a_recall, a_f1 = get_f1(gold_labels_ids, adacad_ids)
print(f"{'AdaCAD':<15} {a_prec:<10.4f} {a_recall:<10.4f} {a_f1:<10.4f}")
