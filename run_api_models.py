from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import time
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
    # ("precise", "en"): "PreciseWikiQA_EN.xlsx",
    # ("precise", "gu"): "PreciseWikiQA_GU.xlsx",
    # ("precise", "hi"): "PreciseWikiQA_HI.xlsx",
    ("precise", "en"): "PreciseWiki_Tamil_EN.xlsx",
    ("precise", "ta"): "PreciseWiki_Tamil_TA.xlsx",
    ("precise", "hi"): "PreciseWiki_Tamil_HI.xlsx",
    ("nonexistent", "en"): "NonExistent_EN.xlsx",
    ("nonexistent", "gu"): "NonExistent_GU.xlsx",
    ("nonexistent", "hi"): "NonExistent_HI.xlsx",
}

# Prompt templates for dataset types
PROMPT_TEMPLATES = {
    ("precise", "en"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Gujarati, answer in Gujarati. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: No Information Available\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("precise", "gu"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Gujarati, answer in Gujarati. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: No Information Available/ માહિતી ઉપલબ્ધ નથી\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("precise", "hi"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Hindi, answer in Hindi. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: कोई जानकारी उपलब्ध नहीं है\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("precise", "ta"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Tamil, answer in Tamil. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: தகவல் இல்லை\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("nonexistent", "en"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Gujarati, answer in Gujarati. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: No Information Available/ માહિતી ઉપલબ્ધ નથી\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("nonexistent", "gu"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Gujarati, answer in Gujarati. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: No Information Available/ માહિતી ઉપલબ્ધ નથી\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("nonexistent", "hi"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Hindi, answer in Hindi. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: No Information Available/ कोई जानकारी उपलब्ध नहीं है\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("nonexistent", "ta"): ( 
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Tamil, answer in Tamil. "
        "If the question is in English, answer in English. "
        "If no factual answer exists, reply exactly: தகவல் இல்லை\n\n"
        "Question: {question}\nAnswer:"
    ),
}

def load_dataset(ds_dir, dtype, lang):
    key = (dtype, lang)
    if key not in DATASET_FILES:
        raise ValueError(f"Unsupported dataset/language pair: {key}")

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

        df = df.rename(columns={q_col: "question"})

        if a_col not in df.columns:
            if dtype == "nonexistent":
                df["gold_answer"] = "માહિતી ઉપલબ્ધ નથી"
            else:
                raise ValueError(f"Expected '{a_col}' in Gujarati file. Found: {list(df.columns)}")
        else:
            df = df.rename(columns={a_col: "gold_answer"})

        if d_col in df.columns:
            df = df.rename(columns={d_col: "domain"})
        else:
            df["domain"] = "unknown"

    # ---------------- HINDI ----------------
    elif lang == "hi":
        # Expected Hindi column names after your translation step
        q_candidates = ["question_hindi", "question"]
        a_candidates = ["answer_hindi", "gold_answer", "answer"]
        d_candidates = ["domain_hindi", "domain", "category"]

        q_col = next((c for c in q_candidates if c in df.columns), None)
        a_col = next((c for c in a_candidates if c in df.columns), None)
        d_col = next((c for c in d_candidates if c in df.columns), None)

        if q_col is None:
            raise ValueError(f"Expected Hindi question column. Found: {list(df.columns)}")

        df = df.rename(columns={q_col: "question"})

        if a_col is None:
            if dtype == "nonexistent":
                df["gold_answer"] = "कोई जानकारी उपलब्ध नहीं है"
            else:
                raise ValueError(f"Expected Hindi answer column. Found: {list(df.columns)}")
        else:
            df = df.rename(columns={a_col: "gold_answer"})

        if d_col is not None:
            df = df.rename(columns={d_col: "domain"})
        else:
            df["domain"] = "unknown"
    
    # ---------------- TAMIL ----------------
    elif lang == "ta":
        q_candidates = ["question_tamil", "question"]
        a_candidates = ["answer_tamil", "gold_answer", "answer"]
        d_candidates = ["domain_tamil", "domain", "category"]

        q_col = next((c for c in q_candidates if c in df.columns), None)
        a_col = next((c for c in a_candidates if c in df.columns), None)
        d_col = next((c for c in d_candidates if c in df.columns), None)

        if q_col is None:
            raise ValueError(f"Expected Tamil question column. Found: {list(df.columns)}")

        df = df.rename(columns={q_col: "question"})

        if a_col is None:
            if dtype == "nonexistent":
                df["gold_answer"] = "தகவல் இல்லை"
            else:
                raise ValueError(f"Expected Tamil answer column. Found: {list(df.columns)}")
        else:
            df = df.rename(columns={a_col: "gold_answer"})

        if d_col is not None:
            df = df.rename(columns={d_col: "domain"})
        else:
            df["domain"] = "unknown"
    else:
        raise ValueError(f"Unsupported language: {lang}")

    df.insert(0, "id", range(len(df)))
    return df[["id", "question", "gold_answer", "domain"]]

def call_api(model_tag, prompt):
    """Unified API caller for GPT, Claude, and Together AI models."""
    # --- GPT-5 ---
    if model_tag == "gpt-5":
        res = client_openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0
        )
        return res.choices[0].message.content.strip()

    # --- Claude ---
    if model_tag in ["claude-sonnet-4-6", "claude-3-sonnet"]:
        res = client_claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content[0].text.strip()

    return "ERROR: Unknown model"

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--dataset_type", required=True, choices=["precise", "nonexistent"])
    parser.add_argument("--lang", required=True, choices=["en", "gu", "hi", "ta"])
    args = parser.parse_args()

    print("=== Starting API model:", args.model_tag)
    print("=== Dataset:", args.dataset_type, "| Language:", args.lang)
    print("=== Loading dataset...")

    base = Path(".")
    ds_df = load_dataset(base / "datasets", args.dataset_type, args.lang)

    print("Loaded", len(ds_df), "rows")
    print("Columns:", list(ds_df.columns))

    out_dir = base / "outputs" / args.dataset_type / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.model_tag}.jsonl"
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

    template = PROMPT_TEMPLATES.get((args.dataset_type, args.lang))
    if template is None:
        raise ValueError(f"No prompt template found for {(args.dataset_type, args.lang)}")

    print("=== Beginning inference...")

    with out_path.open("a", encoding="utf-8") as fout:
        for _, row in tqdm(ds_df.iterrows(), total=len(ds_df)):
            rid = int(row["id"])

            if rid in done_ids:
                continue

            if "{question}" in template:
                prompt = template.format(question=row["question"])
            else:
                prompt = template + f"\nQuestion: {row['question']}\nAnswer:"

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