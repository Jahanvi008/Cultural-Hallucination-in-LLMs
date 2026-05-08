from dotenv import load_dotenv
load_dotenv()

import argparse, json, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from openai import OpenAI
import anthropic
import google.generativeai as genai
from sarvamai import SarvamAI
import os
# import cohere

client_openai = OpenAI()
client_claude = anthropic.Anthropic()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
client_qwen = OpenAI(
    api_key=os.environ.get("QWEN_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
client_openrouter = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
client_deepseek = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
client_sarvam = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))

# client_cohere = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

# --- Dataset file map ---
DATASET_FILES = {
    ("precise", "en"): "IndicWikiQA-Guj_EN.xlsx",
    ("precise", "gu"): "IndicWikiQA-Guj_GU.xlsx",
    ("precise", "hi"): "IndicWikiQA-Guj_HI.xlsx",
    ("nonexistent", "en"): "IndicNonExi-Guj_EN.xlsx",
    ("nonexistent", "gu"): "IndicNonExi-Guj_GU.xlsx",
    ("nonexistent", "hi"): "IndicNonExi-Guj_HI.xlsx",

    ################# TAMIL DATASETS ##################
    ("precise_ta", "en"): "IndicWikiQA-Tam_EN.xlsx",
    ("precise_ta", "hi"): "IndicWikiQA-Tam_HI.xlsx",
    ("precise_ta", "ta"): "IndicWikiQA-Tam_TA.xlsx",
    ("nonexistent_ta", "en"): "IndicNonExi-Tam_EN.xlsx",
    ("nonexistent_ta", "hi"): "IndicNonExi-Tam_HI.xlsx",
    ("nonexistent_ta", "ta"): "IndicNonExi-Tam_TA.xlsx",
}

# Prompt templates for dataset types
PROMPT_TEMPLATES = {
    ("precise", "en"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in English, answer in English. "
        "If no factual answer exists or you don't know the answer, reply exactly: No Information Available\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("precise", "gu"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Gujarati, answer in Gujarati. "
        "If no factual answer exists or you don't know the answer, reply exactly: માહિતી ઉપલબ્ધ નથી\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("precise", "hi"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Hindi, answer in Hindi. "
        "If no factual answer exists or you don't know the answer, reply exactly: कोई जानकारी उपलब्ध नहीं है\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("nonexistent", "en"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in English, answer in English. "
        "If no factual answer exists, OR you don't know the answer reply exactly: No Information Available\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("nonexistent", "gu"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Gujarati, answer in Gujarati. "
        "If no factual answer exists, OR you don't know the answer reply exactly: માહિતી ઉપલબ્ધ નથી\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("nonexistent", "hi"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Hindi, answer in Hindi. "
        "If no factual answer exists, OR you don't know the answer reply exactly: कोई जानकारी उपलब्ध नहीं है\n\n"
        "Question: {question}\nAnswer:"
    ),
    ################## TAMIL DATASETS ##################
    ("precise_ta", "en"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in English, answer in English. "
        "If no factual answer exists or you don't know the answer, reply exactly: No Information Available\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("precise_ta", "ta"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Tamil, answer in Tamil. "
        "If no factual answer exists or you don't know the answer, reply exactly: தகவல் இல்லை\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("precise_ta", "hi"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Hindi, answer in Hindi. "
        "If no factual answer exists or you don't know the answer, reply exactly: कोई जानकारी उपलब्ध नहीं है\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("nonexistent_ta", "en"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in English, answer in English. "
        "If no factual answer exists, OR you don't know the answer reply exactly: No Information Available\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("nonexistent_ta", "hi"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Hindi, answer in Hindi. "
        "If no factual answer exists, OR you don't know the answer reply exactly: कोई जानकारी उपलब्ध नहीं है\n\n"
        "Question: {question}\nAnswer:"
    ),
    ("nonexistent_ta", "ta"): (
        "You are a multilingual factual QA assistant. "
        "Provide a short, precise answer (1–5 words). "
        "If the question is in Tamil, answer in Tamil. "
        "If no factual answer exists, OR you don't know the answer reply exactly: தகவல் இல்லை\n\n"
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
            if dtype == "nonexistent" or dtype == "nonexistent_ta":
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
        # Expected Gujarati column names
        q_candidates = ["question_gujarati", "questions", "Question", "question"]
        a_candidates = ["answer_gujarati", "gold_answer", "answer"]
        d_candidates = ["domain_gujarati", "domain", "category"]

        q_col = next((c for c in q_candidates if c in df.columns), None)
        a_col = next((c for c in a_candidates if c in df.columns), None)
        d_col = next((c for c in d_candidates if c in df.columns), None)


        if q_col not in df.columns:
            raise ValueError(f"Expected '{q_col}' in Gujarati file. Found: {list(df.columns)}")

        df = df.rename(columns={q_col: "question"})

        if a_col not in df.columns:
            if dtype == "nonexistent" or dtype == "nonexistent_ta":
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
        q_candidates = ["question_hindi", "question", "Question", "questions"]
        a_candidates = ["answer_hindi", "gold_answer", "answer"]
        d_candidates = ["domain_hindi", "domain", "category"]

        q_col = next((c for c in q_candidates if c in df.columns), None)
        a_col = next((c for c in a_candidates if c in df.columns), None)
        d_col = next((c for c in d_candidates if c in df.columns), None)

        if q_col is None:
            raise ValueError(f"Expected Hindi question column. Found: {list(df.columns)}")

        df = df.rename(columns={q_col: "question"})

        if a_col is None:
            if dtype == "nonexistent" or dtype == "nonexistent_ta":
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
        q_candidates = ["question_tamil", "question", "Question", "questions"]
        a_candidates = ["answer_tamil", "gold_answer", "answer"]
        d_candidates = ["domain_tamil", "domain", "category"]

        q_col = next((c for c in q_candidates if c in df.columns), None)
        a_col = next((c for c in a_candidates if c in df.columns), None)
        d_col = next((c for c in d_candidates if c in df.columns), None)

        if q_col is None:
            raise ValueError(f"Expected Tamil question column. Found: {list(df.columns)}")

        df = df.rename(columns={q_col: "question"})

        if a_col is None:
            if dtype == "nonexistent" or dtype == "nonexistent_ta":
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
    if model_tag in ["claude-sonnet-4-6"]:
        res = client_claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content[0].text.strip()

    # --- Cohere Command-A ---
    # if model_tag == "command-a":
    #         res = client_cohere.chat(
    #             model="command-a-03-2025",
    #             messages=[{"role": "user", "content": prompt}],
    #             temperature=0.0,
    #             max_tokens=64
    #         )
    #         return res.message.content[0].text.strip()
        
    # --- Qwen ---
    if model_tag in ["qwen-mt-plus"]:
        res = client_qwen.chat.completions.create(
            model="qwen-mt-plus",
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content

    # -- Llama --
    if model_tag in ["llama-4-maverick"]:
        res = client_openrouter.chat.completions.create(
            model="meta-llama/llama-4-maverick",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return res.choices[0].message.content.strip()
    
    # --- Gemini 2.5 Flash ---
    if model_tag == "gemini-2.5-flash":
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()

    # --- DeepSeek-V4 Flash ---
    if model_tag == "deepseek-v4-flash":
        res = client_deepseek.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return res.choices[0].message.content.strip()

    # --- Sarvam-105B ---
    if model_tag == "sarvam-105b":
        res = client_sarvam.chat.completions(
            model="sarvam-105b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return res.choices[0].message.content.strip()
    
    return "ERROR: Unknown model"

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--dataset_type", required=True, choices=["precise", "nonexistent", "precise_ta", "nonexistent_ta"])
    parser.add_argument("--lang", required=True, choices=["en", "gu", "hi", "ta"])
    args = parser.parse_args()

    print("=== Starting API model:", args.model_tag)
    print("=== Dataset:", args.dataset_type, "| Language:", args.lang)
    print("=== Loading dataset...")

    base = Path(".")
    ds_df = load_dataset(base / "datasets", args.dataset_type, args.lang)

    print("Loaded", len(ds_df), "rows")
    print("Columns:", list(ds_df.columns))


    # ---------------- Output folder by culture ----------------
    if args.dataset_type in ["precise", "nonexistent"]:
        culture_output_dir = "model_outputs_gujarati"
    elif args.dataset_type in ["precise_ta", "nonexistent_ta"]:
        culture_output_dir = "model_outputs_tamil"
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    out_dir = base / culture_output_dir / args.dataset_type / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.model_tag}.jsonl"
    done_ids = set()

    if (args.dataset_type, args.lang) not in DATASET_FILES:
        raise ValueError(
            f"Invalid combination: dataset_type={args.dataset_type}, lang={args.lang}. "
            f"Valid combinations are: {list(DATASET_FILES.keys())}"
        )

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