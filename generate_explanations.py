# generate_explanations.py
# Adds multi-level explanations to question JSON produced by the generator.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   python generate_explanations.py \
#       --in questions_asthma.json \
#       --out questions_asthma_explained.json \
#       --specialty "Cardiology" \
#       --topic "Asthma" \
#       --model gpt-5 \
#       --batch-size 10
#
# Input JSON shape (array of objects):
#   { topic_id, type, stem, options, correct_answer }
#
# Output JSON shape (same fields + explanations):
#   {
#     topic_id, type, stem, options, correct_answer,
#     explanation_l2: str,
#     explanation_eli5: str,
#     explanation_points_by_option: { "0": ["..."], ..., "4": ["..."] }
#   }
#
# Notes:
# - Batching reduces overhead cost vs. one call per question. Default: 10 per call.
# - Only fields needed for explanations are sent to the model.
# - The model is instructed to cite guideline names briefly in L2 where relevant (e.g., NICE NG136).

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

MODEL_DEFAULT = "gpt-5"

SYSTEM_PROMPT = (
    "You are a UK medical educator. Write clear, accurate, clinically-aligned explanations for UKMLA-style "
    "Single Best Answer (SBA) questions. Use UK spelling and, when helpful, mention guideline names succinctly "
    "(e.g., NICE NG136) without links. Keep to the requested lengths and formats exactly."
)


def build_user_prompt(specialty: str, topic: str, items_json: str) -> str:
    return f"""
    You will receive a JSON object with an array "questions". Each question has:
        - "idx": integer (stable index to return)
        - "stem": the vignette (no options included)
        - "options": array of 5 strings (A–E)
        - "correct_answer": integer 0–4 indicating the correct option's index in "options"

    TASK: For each question, return a STRICT JSON object with an array "explanations" of the same length. Each item must have:
        - "idx": the same integer you received
        - "explanation_points_by_option": object that maps each option index as a STRING key ("0".."4") to an array with EXACTLY ONE short bullet (one sentence):
            • For the CORRECT option: one quick reason why it is correct in this vignette.
            • For every INCORRECT option: one concise reason why that option is wrong for this vignette.
        - "explanation_l2": a brief paragraph (3–6 sentences) explaining the answer; when appropriate, include a short guideline cue by name only (e.g., NICE NG136, Resus Council ALS) — optional if not needed
        - "explanation_eli5": 1–2 very simple sentences explaining the same idea in plain language

    STYLE:
        - Be precise and clinically correct for UK practice (NICE/BNF/Resus Council UK framing).
        - Use neutral, student-friendly tone. Keep bullets crisp and non-redundant.
        - No markdown, no code fences, no extra keys or commentary.

    CONTEXT (may guide tone and scope): specialty="{specialty}", topic="{topic}".

    INPUT JSON:
    {items_json}

    OUTPUT FORMAT (STRICT):
    {{
        "explanations": [
        {{
            "idx": <int>,
            "explanation_points_by_option": {{
                "0": ["…"],
                "1": ["…"],
                "2": ["…"],
                "3": ["…"],
                "4": ["…"]
            }},
            "explanation_l2": "short paragraph...",
            "explanation_eli5": "one or two very simple sentences"
        }},
        ...
        ]
    }}
    """


def call_openai(model: str, system_prompt: str, user_prompt: str, temperature: float = 1, max_retries: int = 3) -> str:
    if OpenAI is None:
        raise RuntimeError(
            "OpenAI SDK not installed. Run: pip install openai")
    client = OpenAI()
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2.0


def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\\n", "", text)
    text = re.sub(r"\\n```$", "", text)
    return text.strip()


def parse_json_safely(raw: str) -> Dict[str, Any]:
    raw = strip_code_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end+1])
        raise


def chunk(lst: List[Any], size: int) -> List[List[Any]]:
    return [lst[i:i+size] for i in range(0, len(lst), size)]


def main():
    ap = argparse.ArgumentParser(
        description="Generate multi-level explanations for SBA questions and merge into JSON.")
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Input questions JSON (from generator script).")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Output JSON with explanations merged.")
    ap.add_argument("--specialty", default="",
                    help="Optional: specialty name for context (e.g., Cardiology).")
    ap.add_argument("--topic", default="",
                    help="Optional: topic name for context (e.g., Acute Coronary Syndrome).")
    ap.add_argument("--model", default=MODEL_DEFAULT,
                    help="OpenAI model (e.g., gpt-4o, gpt-5, gpt-5-mini).")
    ap.add_argument("--temperature", type=float,
                    default=1, help="Sampling temperature.")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="How many questions per API call.")
    args = ap.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise ValueError(
            "Input file must be a JSON array of question objects.")

    merged = list(questions)

    payload_items = []
    for idx, q in enumerate(questions):
        payload_items.append({
            "idx": idx,
            "stem": q.get("stem", ""),
            "options": q.get("options", []),
            "correct_answer": q.get("correct_answer", ""),
        })

    batches = chunk(payload_items, max(1, args.batch_size))
    idx_to_expl: Dict[int, Dict[str, Any]] = {}

    for b in tqdm(batches, desc="Generating explanations"):
        items_json = json.dumps(
            {"questions": b}, ensure_ascii=False, indent=2)
        prompt = build_user_prompt(args.specialty, args.topic, items_json)
        raw = call_openai(args.model, SYSTEM_PROMPT,
                          prompt, temperature=args.temperature)
        parsed = parse_json_safely(raw)
        expl_list = parsed.get("explanations", [])
        if not isinstance(expl_list, list):
            continue
        for item in expl_list:
            if not isinstance(item, dict):
                continue
            idx = item.get("idx", None)
            if not isinstance(idx, int):
                continue
            pbo = item.get("explanation_points_by_option", {})
            l2 = item.get("explanation_l2", "")
            eli5 = item.get("explanation_eli5", "")
            # Basic validation / shaping
            if not isinstance(pbo, dict):
                pbo = {}
            # normalise keys to strings
            pbo_norm = {}
            for k, v in pbo.items():
                key = str(k)
                if isinstance(v, list):
                    # keep max 1 bullet per option as instructed
                    vals = [str(x).strip() for x in v if str(x).strip()][:1]
                else:
                    vals = [str(v).strip()][:1]
                pbo_norm[key] = vals
            l2 = str(l2).strip()
            eli5 = str(eli5).strip()
            idx_to_expl[idx] = {
                "explanation_points_by_option": pbo_norm,
                "explanation_l2": l2,
                "explanation_eli5": eli5,
            }

    # Merge back into original questions
    for idx, q in enumerate(merged):
        expl = idx_to_expl.get(idx)
        if not expl:
            continue
        q["explanation_points_by_option"] = expl["explanation_points_by_option"]
        q["explanation_l2"] = expl["explanation_l2"]
        q["explanation_eli5"] = expl["explanation_eli5"]

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote explanations for {len(idx_to_expl)} / {len(merged)} questions -> {args.out_path}")


if __name__ == "__main__":
    main()
