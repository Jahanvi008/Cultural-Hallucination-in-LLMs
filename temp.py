import os
import json
import pandas as pd

def get_column(df, keywords):
    """Helper to detect columns ignoring case."""
    for col in df.columns:
        if str(col).strip().lower() in keywords:
            return col
    return None

def main():
    base_dir = "/Users/jaathavanranjanathan/school/4055U/Cultural-Hallucination-in-LLMs"
    models_out_dir = os.path.join(base_dir, "model_outputs_tamil/precise_ta")

    lang_map = {
        "en": "datasets/IndicWikiQA-Tam_EN.xlsx",
        "hi": "datasets/IndicWikiQA-Tam_HI.xlsx",
        "ta": "datasets/IndicWikiQA-Tam_TA.xlsx"
    }

    # The exact IDs you want to drop from the OLD format
    ids_to_remove = {26, 72}

    for lang, excel_filename in lang_map.items():
        excel_path = os.path.join(base_dir, excel_filename)
        jsonl_dir = os.path.join(models_out_dir, lang)
        
        if not os.path.exists(excel_path) or not os.path.exists(jsonl_dir):
            print(f"Skipping {lang}: Directory or excel file not found.")
            continue
            
        print(f"\n--- Processing Language: {lang.upper()} ---")
        
        # 1. Read the Excel File
        df = pd.read_excel(excel_path)
        
        # Determine the columns for questions and answers
        q_col = get_column(df, ['question', 'questions', 'கேள்வி', 'प्रश्न'])
        a_col = get_column(df, ['gold_answer', 'gold answer', 'answer', 'expected answer'])
        
        if not q_col or not a_col:
            print(f"Error: Could not find Question or Answer columns in {excel_filename}. Found: {df.columns.tolist()}")
            continue
            
        # Parse into a clean sequential list of dicts (automatically dropping empty rows)
        excel_data = []
        for _, row in df.iterrows():
            q = str(row[q_col]).strip()
            a = str(row[a_col]).strip()
            if q == 'nan' or not q:
                continue
            excel_data.append({"question": q, "gold_answer": a})

        # 2. Iterate through and update each JSONL file
        for file_name in os.listdir(jsonl_dir):
            if not file_name.endswith(".jsonl"):
                continue
                
            file_path = os.path.join(jsonl_dir, file_name)
            
            # Read existing JSONL and drop the explicit IDs
            filtered_items = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    if item.get("id") not in ids_to_remove:
                        filtered_items.append(item)
                        
            # Check length matches (to ensure your Excel and JSONL logic line up)
            if len(filtered_items) != len(excel_data):
                print(f"Warning for {file_name}: Filtered JSONL has {len(filtered_items)} items, but Excel has {len(excel_data)} items.")
            
            # Map the new Excel data onto the filtered JSON items and reassign sequential IDs
            updated_items = []
            for new_idx, item in enumerate(filtered_items):
                if new_idx < len(excel_data):
                    item["id"] = new_idx
                    item["question"] = excel_data[new_idx]["question"]
                    item["gold_answer"] = excel_data[new_idx]["gold_answer"]
                    updated_items.append(item)
                else:
                    break # Stop if we run out of Excel data

            # Write the cleaned data back to the JSONL
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in updated_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
            print(f"Successfully updated {file_name}: Generated {len(updated_items)} items (IDs 0 to {len(updated_items)-1}).")

if __name__ == "__main__":
    main()