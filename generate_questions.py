# generate_questions.py
# SynapseUK Question Generator — MCQ-only (A–E), JSON output matching DB schema
#
# Usage examples:
#   export OPENAI_API_KEY=sk-...
#   python generate_questions.py \
#       --specialty "Cardiology" \
#       --topic "Asthma" \
#       --topic-id "a8769dfc-2521-49be-99c7-f158816dce46" \
#       --n 20 \
#       --out questions_asthma.json
#
# Requirements:
#   pip install openai python-dotenv tqdm
#
# Notes:
# - Outputs a JSON array of objects with fields: id, topic_id, type, stem, correct_answer
# - 'stem' includes the vignette and the five options A–E (one line per option).
# - 'correct_answer' is a single uppercase letter in {'A','B','C','D','E'}.
# - Easily adapt: pass different --n, --topic, --specialty, --topic-id.
# - This script focuses ONLY on generating questions (no explanations, no dedupe).

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import random

from dotenv import load_dotenv
from tqdm import tqdm

# OpenAI SDK (>=1.0)
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None

load_dotenv()

MODEL_DEFAULT = "gpt-5"

SYSTEM_PROMPT = (
    "You are a UK medical educator creating high-quality Single Best Answer (SBA) multiple-choice "
    "questions for SynapseUK's clinical years platform. Follow UK practice, align with UKMLA style, "
    "and prefer current NICE/BNF/Resuscitation Council UK guidance terminology. Use UK spelling. "
    "Exactly one best answer per item. No images."
)


def build_user_prompt(specialty: str, topic: str, n: int) -> str:
    return f"""        TASK: Create {n} UNIQUE Single Best Answer (SBA) questions for the topic "{topic}" in the specialty "{specialty}".

    AUDIENCE & LEVEL:
    - UK medical students (clinical years), MLA/AKT-aligned difficulty (final-year range).

    STYLE & SCOPE REQUIREMENTS:
    - Each question is a short clinical vignette with five options labelled A–E.
    - Exactly ONE correct option per question.
    - Keep vignettes clinically realistic.
    - Use a healthy mix across diagnosis, investigations/interpretation (including ECGs/labs described in text), acute management, chronic management, complications, and safety-netting.
    - Include at least one investigation to interpret and at least one red-flag/emergency scenario.
    - When appropriate to the topic, include one paediatric or pregnancy presentation.
    - Use UK spelling and UK guideline framing (NICE/BNF/Resus Council UK) when relevant.
    - Do NOT include explanations, rationales, tips, or references — QUESTIONS ONLY.
    - Vary the position of the correct option across questions; do NOT always place it first.
    - Avoid trick wording, double-negatives, or ambiguous stems. Ensure options are mutually exclusive and at the same logical level.

    DIFFICULTY MIX:
    - Target approximate distribution across the set: 40% Easy, 45% Moderate, 15% Hard.
    - Do NOT label difficulty in the output; this is for internal balance only.

    OUTPUT FORMAT (STRICT):
    Return a single JSON array with exactly {n} objects. Each object MUST have only these keys:
    — "vignette": string (the question stem, WITHOUT options)
    — "options": array of exactly 5 short strings (option texts only)
    — "correct_letter": string ("A"–"E")
    Do NOT embed options in the "vignette" field or anywhere outside the "options" array.

    HARD CONSTRAINTS:
    - options length must be exactly 5.
    - Use only characters A–E for correct_letter (uppercase).
    - No markdown, no code fences, no prose outside the JSON array.
    """


def call_openai(model: str, system_prompt: str, user_prompt: str, temperature: float = 1) -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")
    client = OpenAI()
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


def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    text = text.strip()
    return text


def parse_items(raw_json: str) -> List[Dict[str, Any]]:
    raw_json = strip_code_fences(raw_json)
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        start = raw_json.find('[')
        end = raw_json.rfind(']')
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw_json[start:end+1])
        else:
            raise
    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON array.")
    return data


OPTION_LABEL_RE = re.compile(r'^\s*[A-E][\.\)]\s+')


def _clean_option(opt: str) -> str:
    return OPTION_LABEL_RE.sub('', str(opt)).strip()


def letter_to_index(letter: str) -> int:
    s = (letter or "").strip().upper()
    return "ABCDE".find(s)


def validate_and_transform(items: List[Dict[str, Any]], topic_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        vignette = (it.get("vignette") or "").strip()
        options = it.get("options") or []
        correct_letter = (it.get("correct_letter") or "").strip().upper()

        if not vignette:
            continue
        if not isinstance(options, list) or len(options) != 5:
            continue

        orig_idx = letter_to_index(correct_letter)
        if orig_idx < 0 or orig_idx > 4:
            continue

        cleaned_options = [_clean_option(o) for o in options]
        correct_text = cleaned_options[orig_idx]

        shuffled = cleaned_options[:]
        random.shuffle(shuffled)

        try:
            new_idx = shuffled.index(correct_text)
        except ValueError:
            shuffled = cleaned_options
            new_idx = orig_idx

        out.append({
            "topic_id": topic_id,
            "type": "MCQ",
            "stem": vignette,
            "options": shuffled,
            "correct_answer": new_idx
        })
    return out


@dataclass
class Args:
    specialty: str
    topic: str
    topic_id: str
    n: int
    model: str
    temperature: float
    out_path: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="Generate MCQs (A–E) for SynapseUK question bank (JSON output).")
    p.add_argument("--specialty", required=True, help="e.g., Cardiology")
    p.add_argument("--topic", required=True,
                   help="e.g., Acute Coronary Syndrome")
    p.add_argument("--topic-id", required=True,
                   help="UUID of the topic row in DB")
    p.add_argument("--n", type=int, default=20,
                   help="Number of questions to generate")
    p.add_argument("--model", default=MODEL_DEFAULT,
                   help="OpenAI model (default: gpt-4o)")
    p.add_argument("--temperature", type=float,
                   default=1, help="Sampling temperature")
    p.add_argument("--out", dest="out_path",
                   default="questions.json", help="Output JSON file path")
    args = p.parse_args()
    return Args(
        specialty=args.specialty,
        topic=args.topic,
        topic_id=args.topic_id,
        n=args.n,
        model=args.model,
        temperature=args.temperature,
        out_path=args.out_path,
    )


def main():
    args = parse_args()

    # Build and dispatch prompt
    user_prompt = build_user_prompt(args.specialty, args.topic, args.n)
    raw = call_openai(args.model, SYSTEM_PROMPT, user_prompt,
                      temperature=args.temperature)

    print(raw)
    # Parse, validate, transform to DB schema
    items = parse_items(raw)
    records = validate_and_transform(items, args.topic_id)

    if len(records) < args.n:
        print(
            f"Warning: requested {args.n} questions but collected {len(records)} after validation.", file=sys.stderr)

    # Save JSON array
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} questions to {args.out_path}")


if __name__ == "__main__":
    main()
