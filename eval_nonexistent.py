import json

# Path to your JSONL file
# FILE_PATH = "outputs/nonexistent/en/gpt-4o.jsonl"
FILE_PATH = "outputs/nonexistent/gu/gpt-4o.jsonl"

REFUSALS = {
    "માહિતી ઉપલબ્ધ નથી",
    "No Information Available",
    "No Information Available/ માહિતી ઉપલબ્ધ નથી"
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

print(f"Total questions: {total}")
print(f"Non-refusal answers: {non_refusal}")
print(f"Non-refusal percentage: {percentage:.2f}%")
