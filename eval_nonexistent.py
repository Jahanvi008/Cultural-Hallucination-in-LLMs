import json
from pathlib import Path
from datetime import datetime

# Path to your JSONL file
FILE_PATH = "model_outputs_tamil/nonexistent_ta/hi/sarvam-105b.jsonl"

# Report file where results will be saved/appended
REPORT_PATH = "nonexis_evaluation_summary.txt"

REFUSALS = {
    "No Information Available",
    "માહિતી ઉપલબ્ધ નથી",
    "कोई जानकारी उपलब्ध नहीं है",
    "தகவல் இல்லை",
    "जानकारी उपलब्ध नहीं है",
    "No Information Available/ માહિતી ઉપલબ્ધ નથી",
    "No Information Available/ कोई जानकारी उपलब्ध नहीं है",
    "No Information Available/ தகவல் இல்லை",
    "No Information Available/ जानकारी उपलब्ध नहीं है",
    ""
}

total = 0
non_refusal = 0

with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        total += 1

        model_answer = record["model_answer"].strip()

        if model_answer not in REFUSALS:
            non_refusal += 1

percentage = (non_refusal / total) * 100 if total > 0 else 0.0

# Console output
print(f"Total questions: {total}")
print(f"Non-refusal answers: {non_refusal}")
print(f"Non-refusal percentage: {percentage:.2f}%")

# Append results to report file
with open(REPORT_PATH, "a", encoding="utf-8") as report:
    report.write("=" * 60 + "\n")
    report.write(f"File path: {FILE_PATH}\n")
    report.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.write(f"Total questions: {total}\n")
    report.write(f"Non-refusal answers: {non_refusal}\n")
    report.write(f"Non-refusal percentage: {percentage:.2f}%\n\n")