import os
import pandas as pd
import json


percentiles = [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999, 0.999999999, 0.9999999999, 1.0]
item_percenitle_values = {}




script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "bazaar_full_items_ids.json")
output_file_path = os.path.join(script_dir, "label_clip_values.json")
with open(file_path) as f:
    items = json.load(f)
for item_id in items:
    csv_directory = os.path.join(script_dir, "csv files")
    df_labels = pd.read_csv(os.path.join(csv_directory, f"{item_id}_debug_data.csv"), parse_dates=['timestamp'])
    df_labels = df_labels[["entry_label"]]
    percentile_values = df_labels['entry_label'].quantile(percentiles)
    item_percenitle_values[item_id] = percentile_values.to_dict()
with open(output_file_path, 'w') as f:
    json.dump(item_percenitle_values, f, indent=4)

print(f"Label clip values saved to {output_file_path}")
    