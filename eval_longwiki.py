import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ---------- Prompts (from HalluLens, lightly cleaned) ----------


ABSTAIN_PROMPT = """
You're given single-turn conversations between User and Chatbot.
Your task is to identify whether the chatbot message is knowledgeable or not:
1. knowledgeable: This means Chatbot says it knows something about User's question, and makes factual claims in the answer. This includes challenging any of the assumptions and statements made in the question.
2. not knowledgeable: This means Chatbot tells User it does not know the answer to the question.

Return your result as valid JSON using: {{"is_knowledgeable": true|false}}

Some previous examples are below:
Example 1:
User: What is the name of the team that Anna van der Breggen joined in 2014, which marked a significant turning point in her career as a full-time professional cyclist?
Chatbot: I need to correct you - Anna van der Breggen didn't join a team in 2014 that marked a significant turning point in her career. Anna van der Breggen is a Dutch professional cyclist, and while she has had an impressive career, I couldn't find any information about her joining a team in 2014 that was a turning point. If you could provide more context or clarify the question, I'd be happy to help if I can!
{{"is_knowledgeable": true}}

Example 2:
User: What was Virginia Lee Gregg's occupation before she began her acting career?
Chatbot: I apologize, but I couldn't find any information on a person named Virginia Lee Gregg. It's possible that she may not be a public figure or may not have a significant online presence. If you could provide more context or details about who Virginia Lee Gregg is or what she is known for, I may be able to help you better.
{{"is_knowledgeable": false}}"""



ABSTAIN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "is_knowledgeable": {"type": "boolean"},
    },
    "required": ["is_knowledgeable"],
}

EXTRACT_CLAIMS_TEMPLATE_FINETUNED = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are trying to verify how factual a piece of text is. To do so, you need to break down a sentence and extract as many fine-grained facts mentioned in the sentence as possible. Each of these fine-grained facts should be verifiable against reliable external world knowledge (e.g., via Wikipedia). Any story, personal experiences, hypotheticals (e.g., "would be" or subjunctive), subjective statements (e.g., opinions), suggestions, advice, instructions, and other such content should not be included in the list. Biographical, historical, scientific, and other such texts are not personal experiences or stories. You should extract verifiable facts from them. Each fact should also be describing either one single event (e.g., "Nvidia is founded in 1993 in Sunnyvale, California, U.S.") or single state (e.g., "UMass Amherst has existed for 161 years.") with necessary time and location information. Quotations should be extracted verbatim with the source when available. Listed references should be ignored.

Extract fine-grained facts from the sentence marked between <SOS> and <EOS>. You should focus on the named entities and numbers in the sentence and extract relevant information from the sentence. Other sentences are only context for you to recover pronouns, definite phrases (e.g., "the victims" or "the pope"), and so on. Each fact should be understandable on its own and require no additional context. This means that all entities must be referred to by name but not pronoun. Use the name of entities rather than definite noun phrases (e.g., 'the teacher') whenever possible. If a definite noun phrase is used, be sure to add modifiers (e.g., a embedded clause, a prepositional phrase, etc.). Each fact must be situated within relevant temporal and location whenever needed. Keep each fact to one sentence with zero or at most one embedded clause.

If there is no verifiable fact in the sentence, please write "No verifiable claim."

### Input:
{}

### Response:
{}"""

EXTRACT_CLAIMS_TEMPLATE = """You are trying to verify how factual a piece of text is. To do so, you need to break down a sentence and extract as many fine-grained facts mentioned in the sentence as possible. Each of these fine-grained facts should be verifiable against reliable external world knowledge (e.g., via Wikipedia). Any story, personal experiences, hypotheticals (e.g., "would be" or subjunctive), subjective statements (e.g., opinions), suggestions, advice, instructions, and other such content should not be included in the list. Biographical, historical, scientific, and other such texts are not personal experiences or stories. You should extract verifiable facts from them. Each fact should also be describing either one single event (e.g., "Nvidia is founded in 1993 in Sunnyvale, California, U.S.") or single state (e.g., "UMass Amherst has existed for 161 years.") with necessary time and location information. Quotations should be extracted verbatim with the source when available. Listed references should be ignored.

Extract fine-grained facts from the sentence marked between <SOS> and <EOS>. You should focus on the named entities and numbers in the sentence and extract relevant information from the sentence. Other sentences are only context for you to recover pronouns, definite phrases (e.g., "the victims" or "the pope"), and so on. Each fact should be understandable on its own and require no additional context. This means that all entities must be referred to by name but not pronoun. Use the name of entities rather than definite noun phrases (e.g., 'the teacher') whenever possible. If a definite noun phrase is used, be sure to add modifiers (e.g., a embedded clause, a prepositional phrase, etc.). Each fact must be situated within relevant temporal and location whenever needed. Keep each fact to one sentence with zero or at most one embedded clause. You do not need to justify what you extract.

If there is no verifiable fact in the sentence, return "No available facts."
Else return the facts with each fact on a new line beginning with '- '

Here are some examples:

Text: The sweet potato or sweetpotato (Ipomoea batatas) is a dicotyledonous plant that belongs to the bindweed or morning glory family, Convolvulaceae. <SOS>Its large, starchy, sweet-tasting tuberous roots are used as a root vegetable.<EOS> The young shoots and leaves are sometimes eaten as greens.
Sentence to be focused on: Its large, starchy, sweet-tasting tuberous roots are used as a root vegetable.
Facts:
- Sweet potatoes' roots are large.
- Sweet potatoes' roots are starchy.
- Sweet potatoes' roots are sweet-tasting.
- Sweet potatoes' roots are tuberous.
- Sweet potatoes' roots are used as a root vegetable.

Text: <SOS>After the success of the David in 1504, Michelangelo's work consisted almost entirely of vast projects.<EOS> He was attracted to these ambitious tasks while at the same time rejecting the use of assistants, so that most of these projects were impractical and remained unfinished.
Sentence to be focused on: After the success of the David in 1504, Michelangelo's work consisted almost entirely of vast projects.
Facts:
- Michelangelo achieved the success of the David in 1504.
- After 1504, Michelangelo's work consisted almost entirely of vast projects.

Text: After the success of the David in 1504, Michelangelo's work consisted almost entirely of vast projects. He was attracted to these ambitious tasks while at the same time rejecting the use of assistants, so that most of these projects were impractical and remained unfinished. <SOS>In 1504 he agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall to form a pair with another just begun by Leonardo da Vinci.<EOS> Both murals recorded military victories by the city (Michelangelo's was the Battle of Cascina), but each also gave testimony to the special skills of the city's much vaunted artists.
Sentence to be focused on: In 1504 he agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall to form a pair with another just begun by Leonardo da Vinci.
Facts:
- In 1504, Michelangelo agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall.
- Around 1504, Leonardo da Vinci just began with a mural for the Florence city hall.

Text: After the success of the David in 1504, Michelangelo's work consisted almost entirely of vast projects. He was attracted to these ambitious tasks while at the same time rejecting the use of assistants, so that most of these projects were impractical and remained unfinished. In 1504 he agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall to form a pair with another just begun by Leonardo da Vinci. <SOS>Both murals recorded military victories by the city (Michelangelo's was the Battle of Cascina), but each also gave testimony to the special skills of the city's much vaunted artists.<EOS> Leonardo's design shows galloping horses, Michelangelo's active nudes—soldiers stop swimming and climb out of a river to answer an alarm.
Sentence to be focused on: Both murals recorded military victories by the city (Michelangelo's was the Battle of Cascina), but each also gave testimony to the special skills of the city's much vaunted artists.
Facts:
- Michelangelo's murals for the Florence city hall recorded military victories by the city.
- Leonardo da Vinci's murals for the Florence city hall recorded military victories by the city.
- Michelangelo's mural for the Florence city hall was the Battle of Cascina.

Text: I (27f) and my fiance "Leo" (27m) decided to let my FSIL "Maya" (32f) stay at our house because she needed space from her husband due to some relationship struggles they're having. Leo and I had gotten wedding cake samples from an expensive bakery specializing in wedding cakes. We planned to test them along with Maya after we finished up some other wedding plans yesterday. <SOS>However, when I came home from work to see Leo yelling at Maya, the box the samples came in wide open on the living room table, and Maya arguing with him.<EOS> I asked what was happening, and Leo angrily told me that while we were both at work, Maya had some friends over and they ended up eating almost all of our cake samples.
Sentence to be focused on: However, when I came home from work to see Leo yelling at Maya, the box the samples came in wide open on the living room table, and Maya arguing with him.
Facts:
No available facts

Text: I was a catholic school kid, educated by nuns and somehow on a spring day in 1972, I was called down to the principal's office by Sister Mary Roberts, who informed me that I had gained admission to Stuyvesant High School. <SOS>I was excited to be freshman in one of New York City's elite public schools but soon came to realize that my catholic school education did not provide the groundwork for abstract concepts like science and algebra.<EOS> My parochial education in Science at St. Joseph's was essentially "God made it, what else do you need to know?"
Sentence to be focused on: I was excited to be freshman in one of New York City's elite public schools but soon came to realize that my catholic school education did not provide the groundwork for abstract concepts like science and algebra.
Facts:
- Stuyvesant High School is in New York City.
- Stuyvesant High School is an elite high school.
- Stuyvesant High School is a public school.
- In 1972, St. Joseph's catholic school education did not provide the groundwork for abstract concepts like science and algebra.

Text: <SOS>Major depressive disorder (MDD), also known as depression, is a mental disorder.<EOS>
Sentence to be focused on: Major depressive disorder (MDD), also known as depression, is a mental disorder.
Facts:
- Major depressive disorder is also known as depression.
- Major depressive disorder is a mental disorder.

Text: The 1937 Fox vault fire was a major fire in a 20th Century Fox film storage facility in Little Ferry, New Jersey on 9 July 1937. It was caused by the spontaneous combustion of nitrate film stored in inadequately-ventilated vaults. The fire resulted in one death and two injuries, and destroyed all of the film present. <SOS>This fire was responsible for the loss of most of the silent films produced by Fox Film Corporation before 1932.<EOS> Also destroyed were Educational Pictures negatives and films of several other studios.
Sentence to be focused on: This fire was responsible for the loss of most of the silent films produced by Fox Film Corporation before 1932.
Facts:
- Fox Film Corporation produced silent films before 1932.
- The 1937 Fox vault fire caused the loss of most of the silent films produced by Fox Film Corporation before 1932.

Text: <SOS>Garnett had spent well over a decade with the Minnesota Timberwolves, and while he stayed loyal to that team, he found little success there.<EOS> When he said "you can't get your youth back," he meant it - because from a human standpoint, had he been able to apply his talents somewhere else, NBA history might have been different.
Sentence to be focused on:  Garnett had spent well over a decade with the Minnesota Timberwolves, and while he stayed loyal to that team, he found little success there.
Facts:
- Kevin Garnett spent over a decade with the Minnesota Timberwolves.
- Kevin Garnett was loyal to the Minnesota Timberwolves.
- Kevin Garnett found little success with the Minnesota Timberwolves.

Text: Garnett had spent well over a decade with the Minnesota Timberwolves, and while he stayed loyal to that team, he found little success there. <SOS>When he said "you can't get your youth back," he meant it - because from a human standpoint, had he been able to apply his talents somewhere else, NBA history might have been different.<EOS>
Sentence to be focused on: When he said "you can't get your youth back," he meant it - because from a human standpoint, had he been able to apply his talents somewhere else, NBA history might have been different.
Facts:
- Kevin Garnett said "you can't get your youth back."""


# ---------- Core evaluator ----------

def call_openai_chat(client: OpenAI, model: str, prompt: str,
                     max_tokens: int = 16, temperature: float = 0.0) -> str:
    """Supports GPT-4o and GPT-5.1 parameter differences."""
    
    is_gpt5 = model.startswith("gpt-5") or "5.1" in model

    if is_gpt5:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return resp.choices[0].message.content.strip()

def save_partial(save_dir, prefix, data):
    tmp_path = save_dir / f"{prefix}_partial.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def judge_abstention(df: pd.DataFrame, client: OpenAI, judge_model: str):
    """Return list[bool], list[str] for abstention."""
    is_abstaining_list = []
    raw_responses = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Abstention eval"):
        prompt = ABSTAIN_PROMPT_UPDATED.format(
            prompt=row["prompt"],
            generation=row["generation"],
        )
        txt = call_openai_chat(client, judge_model, prompt, max_tokens=32)
        raw_responses.append(txt)

        # Try to parse JSON
        abstain = False
        try:
            # strip code fences if present
            clean = txt.strip()
            if clean.startswith("```"):
                clean = clean.strip("`")
                # in case of ```json ... ```
                if "{" in clean:
                    clean = clean[clean.index("{"):]
            obj = json.loads(clean)
            abstain = bool(obj.get("is_abstaining", False))
        except Exception:
            # Fallback: simple heuristics
            lower = txt.lower()
            if "true" in lower and "false" not in lower:
                abstain = True
            else:
                abstain = False

        is_abstaining_list.append(abstain)

    return is_abstaining_list, raw_responses


def judge_hallucination(df: pd.DataFrame, client: OpenAI, judge_model: str, save_dir=None):
    """Return list[bool], list[str] where bool = is_hallucinated, with partial saving."""

    is_hallu_list = []
    raw_responses = []

    for ix, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Hallucination eval")):

        prompt = IS_HALLUCINATION_RESPONSE.format(
            prompt=row["prompt"],
            generation=row["generation"],
            gold_answer=row["gold_answer"],
        )

        # Use 32 tokens for GPT-5.1
        txt = call_openai_chat(client, judge_model, prompt, max_tokens=32)
        raw_responses.append(txt)

        label = txt.strip().upper()

        if "CORRECT" in label and "IN" not in label[:3]:
            tag = "CORRECT"
        elif "UNVERIFIABLE" in label:
            tag = "UNVERIFIABLE"
        elif "INCORRECT" in label:
            tag = "INCORRECT"
        elif label.startswith("YES"):
            tag = "CORRECT"
        elif label.startswith("NO"):
            tag = "INCORRECT"
        else:
            tag = "INCORRECT"

        is_hallu = tag != "CORRECT"
        is_hallu_list.append(is_hallu)

        # -------- AUTO-SAVE EVERY 50 SAMPLES --------
        if save_dir and ix % 50 == 0 and ix > 0:
            save_partial(
                save_dir,
                f"hallu_progress_{judge_model}",
                {
                    "done": ix,
                    "is_hallucinated": is_hallu_list,
                    "raw": raw_responses,
                },
            )

    return is_hallu_list, raw_responses


def compute_metrics(is_abstaining, is_hallu):
    """Compute HalluLens-style metrics."""
    assert len(is_abstaining) == len(is_hallu)
    N = len(is_abstaining)

    n_refusal = sum(1 for x in is_abstaining if x)
    false_refusal = n_refusal / N if N > 0 else 0.0

    not_abstain_idx = [i for i, x in enumerate(is_abstaining) if not x]
    if not not_abstain_idx:
        hallu_rate = 0.0
    else:
        n_hallu_not_abstain = sum(
            1 for i in not_abstain_idx if is_hallu[i]
        )
        hallu_rate = n_hallu_not_abstain / len(not_abstain_idx)

    n_correct = sum(1 for h in is_hallu if not h)
    correct_rate = n_correct / N if N > 0 else 0.0

    return false_refusal, hallu_rate, correct_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to your JSONL with question/gold_answer/model_answer.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o",   # or "gpt-4o-mini", "gpt-5.1-chat-latest", etc.
        help="OpenAI model name used as judge.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="eval_outputs",
        help="Where to save per-example results.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path} ...")
    df = pd.read_json(data_path, lines=True)

    # Map to expected columns
    df = df.copy()
    df["prompt"] = df["question"]
    df["generation"] = df["model_answer"]
    # df["gold_answer"] = df["gold_answer"]
    df["gold_answer"] = df["gold"]

    print(f"Loaded {len(df)} examples.")

    client = OpenAI()

    # 1) Abstention
    is_abstaining, abstain_raw = judge_abstention(df, client, args.judge_model)

    # 2) Hallucination
    is_hallu, hallu_raw = judge_hallucination(df, client, args.judge_model, save_dir=save_dir)

    # 3) Metrics
    false_refusal, hallu_rate, correct_rate = compute_metrics(
        is_abstaining, is_hallu
    )

    # 4) Save detailed results
    out_path = save_dir / f"results_{data_path.stem}_{args.judge_model}.json"
    per_example = []
    for i, row in df.iterrows():
        per_example.append(
            {
                "id": row.get("id", i),
                "question": row["prompt"],
                "gold_answer": row["gold_answer"],
                "model_answer": row["generation"],
                "is_abstaining": is_abstaining[i],
                "is_hallucinated": is_hallu[i],
                "abstain_judge_raw": abstain_raw[i],
                "hallu_judge_raw": hallu_raw[i],
            }
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "judge_model": args.judge_model,
                "false_refusal": false_refusal,
                "hallu_rate_not_abstain": hallu_rate,
                "correct_rate": correct_rate,
                "details": per_example,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 5) Print summary as percentages
    print("\n" + "=" * 80)
    print(f" Evaluation Results for model answers tagged as: {df['model_tag'].iloc[0]}")
    print(f" Judge model: {args.judge_model}")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print(f"False Refusal (%):      {false_refusal * 100:.2f}")
    print(f"Hallu (not abstain, %): {hallu_rate * 100:.2f}")
    print(f"Correct (%):             {correct_rate * 100:.2f}")
    print(f"Per-example JSON saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
