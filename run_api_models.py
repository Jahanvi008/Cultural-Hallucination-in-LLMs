from dotenv import load_dotenv
load_dotenv()

import argparse, json, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from openai import OpenAI
import anthropic
import requests
import os
client_openai = OpenAI()
client_claude = anthropic.Anthropic()

# --- Dataset file map ---
DATASET_FILES = {
    ("precise", "en"): "PreciseWikiQA_EN.xlsx",
    ("precise", "gu"): "PreciseWikiQA_GU.xlsx",
    ("longwiki", "en"): "LongWikiQA_EN.xlsx",
    ("longwiki", "gu"): "LongWikiQA_GU.xlsx",
    ("nonexistent", "en"): "NonExistent_EN.xlsx",
    ("nonexistent", "gu"): "NonExistent_GU.xlsx",
}

# Prompt templates for dataset types
PROMPT_TEMPLATES = {
    ("precise", "en"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Gujarati, answer in Gujarati. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: No Information Available\n\nQuestion: {question}\nAnswer:"
    ),
    ("precise", "gu"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Gujarati, answer in Gujarati. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: No Information Available/ માહિતી ઉપલબ્ધ નથી\n\nQuestion: {question}\nAnswer:"
    ),
    ("longwiki_answer", "en"): (
        "I would like you to act as a factual long-form answer generator for questions related to Gujarati culture.\n"
        "Your goal is to produce one detailed, factual answer grounded strictly in a relevant Wikipedia article.\n\n"
        "You should identify one Wikipedia article that directly supports the user’s question — for example, related to Gujarati food, festivals, geography, politics, holidays, education, family life, or traditional practices — and use only that article as your factual source.\n\n"
        "Requirements for the Answer:\n\n"
        "1. The answer must be fully supported by the selected Wikipedia article, without using external knowledge or assumptions.\n\n"
        "2. The answer must be written in clear English and contain multiple sentences (≥2), providing factual detail and depth.\n\n"
        "3. The answer should be objective, neutral, and descriptive, avoiding interpretations, opinions, or invented information.\n\n"
        "4. Use transliterated Gujarati terms only if they appear in the Wikipedia article or are explicitly justified by it.\n\n"
        "5. Do not include citations, bullet points, or meta-commentary — output only the final factual answer.\n"
        "6. If no factual answer exists, reply exactly: No Information Available/ માહિતી ઉપલબ્ધ નથી\n\nQuestion: {question}\nAnswer:"
    ),
    ("longwiki", "gu"): (
        "I would like you to act as a factual long-form answer generator for questions related to Gujarati culture.\n"
        "Your goal is to produce one detailed, factual answer grounded strictly in a relevant Wikipedia article.\n\n"
        "You should identify one Wikipedia article that directly supports the user’s question — for example, related to Gujarati food, festivals, geography, politics, holidays, education, family life, or traditional practices — and use only that article as your factual source.\n\n"
        "Requirements for the Answer:\n\n"
        "1. The answer must be fully supported by the selected Wikipedia article, without using external knowledge or assumptions.\n\n"
        "2. The answer must be written in clear Gujarati and contain multiple sentences (≥2), providing factual detail and depth.\n\n"
        "3. The answer should be objective, neutral, and descriptive, avoiding interpretations, opinions, or invented information.\n\n"
        "4. Use transliterated Gujarati terms only if they appear in the Wikipedia article or are explicitly justified by it.\n\n"
        "5. Do not include citations, bullet points, or meta-commentary — output only the final factual answer.\n"
        "6. If no factual answer exists, reply exactly: No Information Available/ માહિતી ઉપલબ્ધ નથી\n\nQuestion: {question}\nAnswer:"
    ),
    ("nonexistent", "en"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Gujarati, answer in Gujarati. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: No Information Available/ માહિતી ઉપલબ્ધ નથી\n\nQuestion: {question}\nAnswer:" 
    ),
    ("nonexistent", "gu"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Gujarati, answer in Gujarati. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: No Information Available/ માહિતી ઉપલબ્ધ નથી\n\nQuestion: {question}\nAnswer:"
    )    
}

# def load_dataset(ds_dir, dtype, lang):
#     key = (dtype, lang)
#     df = pd.read_excel(Path(ds_dir) / DATASET_FILES[key])

#     # Normalize column names
#     df.columns = df.columns.str.lower().str.strip()

#     # English dataset
#     if lang == "en":
#         q_col = "question"
#         a_col = "answer"
#         if q_col not in df or a_col not in df:
#             raise ValueError(f"Expected columns 'Question' and 'Answer' in English file. Found: {df.columns}")

#         df = df.rename(columns={
#             q_col: "question",
#             a_col: "gold_answer"
#         })

#     # Gujarati dataset
#     elif lang == "gu":
#         q_col = "question_gujarati"
#         a_col = "answer_gujarati"
#         if q_col not in df or a_col not in df:
#             raise ValueError(f"Expected Gujarati columns 'Question_gujarati' and 'Answer_gujarati'. Found: {df.columns}")

#         df = df.rename(columns={
#             q_col: "question",
#             a_col: "gold_answer"
#         })

#     # Insert id column
#     df.insert(0, "id", range(len(df)))

#     return df[["id", "question", "gold_answer"]]

def load_dataset(ds_dir, dtype, lang):
    key = (dtype, lang)
    df = pd.read_excel(Path(ds_dir) / DATASET_FILES[key])

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # ---------------- ENGLISH ----------------
    if lang == "en":
        q_candidates = ["question", "questions"]
        a_candidates = ["answer", "gold_answer"]
        d_candidates = ["domain", "category"]

        q_col = next((c for c in q_candidates if c in df.columns), None)
        a_col = next((c for c in a_candidates if c in df.columns), None)
        d_col = next((c for c in d_candidates if c in df.columns), None)

        if q_col is None:
            raise ValueError(f"Missing question column. Found: {list(df.columns)}")

        if a_col is None:
            if dtype == "nonexistent":
                df["gold_answer"] = "No Information Available"
            else:
                raise ValueError(
                    f"Expected an answer column for dataset '{dtype}'. Found: {list(df.columns)}"
                )
        else:
            df = df.rename(columns={a_col: "gold_answer"})

        df = df.rename(columns={q_col: "question"})

        # Handle domain
        if d_col:
            df = df.rename(columns={d_col: "domain"})
        else:
            df["domain"] = "unknown"

    # ---------------- GUJARATI ----------------
    elif lang == "gu":
        q_col = "question_gujarati"
        a_col = "answer_gujarati"
        d_col = "domain_gujarati"

        if q_col not in df.columns:
            raise ValueError(f"Expected '{q_col}' in Gujarati file. Found: {list(df.columns)}")

        # Rename question
        df = df.rename(columns={q_col: "question"})

        # Handle answer
        if a_col not in df.columns:
            if dtype == "nonexistent":
                df["gold_answer"] = "માહિતી ઉપલબ્ધ નથી"
            else:
                raise ValueError(f"Expected '{a_col}' in Gujarati file. Found: {list(df.columns)}")
        else:
            df = df.rename(columns={a_col: "gold_answer"})

        # ✅ Proper domain rename
        if d_col in df.columns:
            df = df.rename(columns={d_col: "domain"})
        else:
            df["domain"] = "unknown"

    # Insert id column
    df.insert(0, "id", range(len(df)))

    return df[["id", "question", "gold_answer", "domain"]]

def call_api(model_tag, prompt):
    """Unified API caller for GPT-4o, Claude, LLaMA 405B, LLaMA 3.3.

    `prompt` should be a fully prepared string (question inserted or template used).
    """
    # --- GPT-5 ---
    if model_tag == "gpt-5":
        res = client_openai.chat.completions.create(
            model="gpt-5", 
            messages=[{"role":"user","content":prompt}],
            temperature=1.0
        )
        return res.choices[0].message.content.strip()

    # --- Claude ---
    if model_tag in ["claude-3-haiku", "claude-3-sonnet"]:
        model_name = "claude-3-haiku-20240307" if model_tag == "claude-3-haiku" else "claude-3-sonnet-20240229"

        res = client_claude.messages.create(
            model=model_name,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content[0].text.strip()

    # --- Together AI: LLaMA models (405B, 3.3-70B) ---
    if model_tag in ["llama-3.1-405b", "llama-3.3-70b"]:
        TOGETHER = "https://api.together.xyz/v1/completions"
        model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct" if model_tag == "llama-3.1-405b" \
                     else "meta-llama/Meta-Llama-3.3-70B-Instruct"

        headers = {"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"}
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.0
        }

        r = requests.post(TOGETHER, json=payload, headers=headers).json()
        return r["output"]["choices"][0]["text"].strip()

    return "ERROR: Unknown model"

def main():
    from dotenv import load_dotenv
    load_dotenv()
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--dataset_type", required=True)
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()

    print("=== Starting API model:", args.model_tag)
    print("=== Dataset:", args.dataset_type, "| Language:", args.lang)
    print("OPENAI_API_KEY =", os.environ.get("OPENAI_API_KEY"))

    base = Path(".")
    print("=== Loading dataset...")

    ds_df = load_dataset(base / "datasets", args.dataset_type, args.lang)
    print("Loaded", len(ds_df), "rows")
    print("Columns:", list(ds_df.columns))

    out_dir = base / "outputs" / args.dataset_type / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.model_tag}.jsonl"
    fout = out_path.open("a", encoding="utf-8")

    done_ids = set()

    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                    done_ids.add(int(obj["id"]))
                except Exception:
                    continue

        print(f"Resuming: {len(done_ids)} already completed.")
        
    print("=== Beginning inference...")

    # Select a prompt template for this dataset/language
    template = PROMPT_TEMPLATES.get((args.dataset_type, args.lang)) or PROMPT_TEMPLATES.get((args.dataset_type, "en"))

    for _, row in tqdm(ds_df.iterrows(), total=len(ds_df)):
        rid = int(row["id"])

        if rid in done_ids:
            continue

        # If the template expects a question placeholder, format it; otherwise use template as-is
        if template and "{question}" in template:
            prompt = template.format(question=row["question"])
        else:
            prompt = template or ("Question: " + row["question"] + "\nAnswer:")

        ans = call_api(args.model_tag, prompt)

        record = {
            "id": int(row["id"]),
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "model_answer": ans,
            "domain": row["domain"],
            "model_tag": args.model_tag,
            "dataset_type": args.dataset_type,
            "lang": args.lang,
        }

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        time.sleep(0.5)

    print("=== Completed ===")

if __name__ == "__main__":
    main()
