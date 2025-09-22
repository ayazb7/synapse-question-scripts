"""
batch_generate.py

End-to-end orchestrator to generate questions, add explanations, and export
per-topic CSV files for upload. It reuses the logic from:
 - generate_questions.py
 - generate_explanations.py
 - db_upload.py

It will:
 1) Load specialties and topics from JSON exports in question_scripts/ (specialties_rows.json, topics_rows.json)
 2) Iterate all topics, excluding those listed in out/done_topics.txt
 2) For each topic, generate N (default 30) questions via OpenAI
 3) Add multi-level explanations in batches
 4) Normalise each question into a CSV row using db_upload's schema helpers

Usage example:
  MAC:
  export OPENAI_API_KEY=sk-...
  python batch_generate.py \
      --out-dir out \
      --n 30 \
      --model gpt-5 \
      --batch-size 10 \
      --temperature 1

  Windows:
  set OPENAI_API_KEY=sk-...
  python batch_generate.py ^
      --out-dir out ^
      --n 30 ^
      --model gpt-5 ^
      --batch-size 10 ^
      --temperature 1

Notes:
 - Requires: pip install openai python-dotenv tqdm
 - Uses the same strict JSON formats enforced by the existing scripts.
 - Continues on errors per topic (logs and skips that topic).
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


# Ensure we can import sibling scripts even when run from project root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

try:
    import generate_questions as gq
    import generate_explanations as ge
    import db_upload as du
except Exception as import_err:  # pragma: no cover
    raise RuntimeError(
        "Failed to import helper scripts from question_scripts/. Run this script from the repo root or ensure PYTHONPATH includes question_scripts/."
    ) from import_err




def _load_json_rows(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        return []
    return []


def load_specialties() -> List[Dict[str, Any]]:
    """Load specialties from specialties_rows.json. Returns empty list if missing."""
    json_path = os.path.join(THIS_DIR, "specialties_rows.json")
    rows = _load_json_rows(json_path)
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "slug": r.get("slug"),
            "created_at": r.get("created_at"),
            "icon_name": r.get("icon_name"),
            "icon_color": r.get("icon_color"),
            "icon_bg_start": r.get("icon_bg_start"),
            "icon_bg_end": r.get("icon_bg_end"),
        })
    return out


def load_topics() -> List[Dict[str, Any]]:
    """Load topics from topics_rows.json. Returns empty list if missing."""
    json_path = os.path.join(THIS_DIR, "topics_rows.json")
    rows = _load_json_rows(json_path)
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "id": r.get("id"),
            "specialty_id": r.get("specialty_id"),
            "name": r.get("name"),
            "slug": r.get("slug"),
            "description": r.get("description"),
            "created_at": r.get("created_at"),
        })
    return out


def load_excluded_topic_names(done_topics_path: str) -> List[str]:
    """Read topic names from done_topics.txt (one per line, possibly quoted and comma-suffixed)."""
    names: List[str] = []
    try:
        with open(done_topics_path, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                # Remove trailing commas
                if s.endswith(","):
                    s = s[:-1].strip()
                # Remove surrounding quotes
                if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
                    s = s[1:-1].strip()
                if s:
                    names.append(s)
    except FileNotFoundError:
        return []
    return names



def sanitise_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "topic"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_questions_for_topic(
    *,
    specialty_name: str,
    topic_name: str,
    topic_id: str,
    n: int,
    model: str,
    temperature: float,
) -> List[Dict[str, Any]]:
    """Use generate_questions helpers to create validated question records."""
    user_prompt = gq.build_user_prompt(specialty_name, topic_name, n)
    raw = gq.call_openai(model, gq.SYSTEM_PROMPT, user_prompt, temperature=temperature)
    items = gq.parse_items(raw)
    records = gq.validate_and_transform(items, topic_id)
    return records


def add_explanations_to_questions(
    *,
    questions: List[Dict[str, Any]],
    specialty_name: str,
    topic_name: str,
    model: str,
    temperature: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Reuse generate_explanations logic to enrich questions in memory."""
    payload_items: List[Dict[str, Any]] = []
    for idx, q in enumerate(questions):
        payload_items.append({
            "idx": idx,
            "stem": q.get("stem", ""),
            "options": q.get("options", []),
            "correct_answer": q.get("correct_answer", ""),
        })

    batches = ge.chunk(payload_items, max(1, int(batch_size)))
    idx_to_expl: Dict[int, Dict[str, Any]] = {}

    for b in tqdm(batches, desc=f"Explanations: {topic_name}"):
        items_json = json.dumps({"questions": b}, ensure_ascii=False, indent=2)
        prompt = ge.build_user_prompt(specialty_name, topic_name, items_json)
        raw = ge.call_openai(model, ge.SYSTEM_PROMPT, prompt, temperature=temperature)
        parsed = ge.parse_json_safely(raw)
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
            l2 = str(item.get("explanation_l2", "")).strip()
            eli5 = str(item.get("explanation_eli5", "")).strip()

            # Normalise pbo keys and values as in script
            if not isinstance(pbo, dict):
                pbo = {}
            pbo_norm: Dict[str, List[str]] = {}
            for k, v in pbo.items():
                key = str(k)
                if isinstance(v, list):
                    # Exactly one concise bullet per option
                    vals = [str(x).strip() for x in v if str(x).strip()][:1]
                else:
                    vals = [str(v).strip()][:1]
                pbo_norm[key] = vals

            idx_to_expl[idx] = {
                "explanation_points_by_option": pbo_norm,
                "explanation_l2": l2,
                "explanation_eli5": eli5,
            }

    # Merge back
    for idx, q in enumerate(questions):
        e = idx_to_expl.get(idx)
        if not e:
            continue
        q["explanation_points_by_option"] = e["explanation_points_by_option"]
        q["explanation_l2"] = e["explanation_l2"]
        q["explanation_eli5"] = e["explanation_eli5"]

    return questions


def build_specialty_map(specialties: List[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    """Return mapping from specialty_id -> (name, slug)."""
    out: Dict[str, Tuple[str, str]] = {}
    for s in specialties:
        sid = s.get("id", "")
        out[sid] = (s.get("name", ""), s.get("slug", ""))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch-generate SBA questions, add explanations, export one CSV.")
    ap.add_argument("--out-dir", default=os.path.join(THIS_DIR, "out"), help="Directory for per-topic CSV outputs and logs.")
    ap.add_argument("--n", type=int, default=30, help="Questions per topic (default 30).")
    ap.add_argument("--model", default="gpt-5", help="OpenAI model (default gpt-5).")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default 1.0).")
    ap.add_argument("--batch-size", type=int, default=10, help="Explanations batch size per API call (default 10).")
    ap.add_argument("--seed", type=int, default=0, help="Optional random seed for shuffling options.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    ensure_dir(args.out_dir)

    # Load data from JSON exports
    specialties = load_specialties()
    topics = load_topics()

    # Per-topic CSVs will be written directly into out-dir
    specialty_map = build_specialty_map(specialties)
    created_at = du.iso_now()
    updated_at = du.iso_now()
    default_is_active = True

    # Timing trackers and logs
    overall_start = time.perf_counter()
    topic_times: List[float] = []
    done_log_path = os.path.join(args.out_dir, "done_topics.txt")

    # Sort topics by specialty then name for predictability
    def _topic_sort_key(t: Dict[str, Any]) -> Tuple[str, str]:
        spec_name = specialty_map.get(t.get("specialty_id", ""), ("", ""))[0]
        return (spec_name or "~", t.get("name", "~"))

    topics_sorted = sorted(topics, key=_topic_sort_key)


    excluded_names = set(n.lower() for n in load_excluded_topic_names(os.path.join(args.out_dir, "done_topics.txt")))
    if not excluded_names:
        default_done_path = os.path.join(THIS_DIR, "out", "done_topics.txt")
        excluded_names = set(n.lower() for n in load_excluded_topic_names(default_done_path))
    processed_counts: List[Tuple[str, str, int]] = []

    for topic in tqdm(topics_sorted, desc="Topics"):
        topic_name = topic.get("name", "")
        topic_slug = sanitise_slug(topic.get("slug", topic_name))
        topic_id = topic.get("id", "")
        spec_id = topic.get("specialty_id", "")
        specialty_name = specialty_map.get(spec_id, ("", ""))[0]

        if topic_name.strip().lower() in excluded_names:
            continue
        if not topic_id or not specialty_name:
            print(f"Skipping topic with missing ids: {topic_name} ({topic_id})")
            continue

        try:
            t_topic_start = time.perf_counter()
            tqdm.write(f"Generating questions for {specialty_name} / {topic_name}...")
            questions = generate_questions_for_topic(
                specialty_name=specialty_name,
                topic_name=topic_name,
                topic_id=topic_id,
                n=int(args.n),
                model=args.model,
                temperature=float(args.temperature),
            )
            t_after_questions = time.perf_counter()

            if not questions:
                print(f"No valid questions produced for {topic_name}; skipping.")
                continue

            # 2) Explanations
            questions_with_expl = add_explanations_to_questions(
                questions=list(questions),
                specialty_name=specialty_name,
                topic_name=topic_name,
                model=args.model,
                temperature=float(args.temperature),
                batch_size=int(args.batch_size),
            )
            t_after_explanations = time.perf_counter()

            # 3) Normalise for CSV and append to master list
            per_topic_rows: List[Dict[str, Any]] = []
            for q in questions_with_expl:
                topic_id_for_row = q.get("topic_id", topic_id)
                row = du.normalise_question(
                    q=q,
                    topic_id=topic_id_for_row,
                    default_is_active=default_is_active,
                    created_at=created_at,
                    updated_at=updated_at,
                )
                per_topic_rows.append(row)

            # 4) Write per-topic CSV
            per_topic_csv_path = os.path.join(args.out_dir, f"{topic_slug}.csv")
            with open(per_topic_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=du.CSV_COLUMNS)
                writer.writeheader()
                for r in per_topic_rows:
                    writer.writerow(r)
            processed_counts.append((specialty_name, topic_name, len(per_topic_rows)))
            tqdm.write(f"[Saved] {topic_slug}.csv ({len(per_topic_rows)} rows)")

            # Timing logs for this topic
            t_topic_end = time.perf_counter()
            q_secs = t_after_questions - t_topic_start
            e_secs = t_after_explanations - t_after_questions
            n_secs = t_topic_end - t_after_explanations
            total_secs = t_topic_end - t_topic_start
            topic_times.append(total_secs)
            tqdm.write(
                f"[Timing] {specialty_name} / {topic_name}: questions {q_secs:.1f}s | explanations {e_secs:.1f}s | normalise {n_secs:.1f}s | total {total_secs:.1f}s"
            )

            # Append to done log
            with open(done_log_path, "a", encoding="utf-8") as logf:
                logf.write(f"{topic_name}\n")

        except Exception as e:  # continue on failure of one topic
            print(f"Error processing '{topic_name}' ({specialty_name}): {e}")
            continue

    # Small summary per specialty/topic
    for spec_name, tname, count in processed_counts:
        print(f" - {spec_name} / {tname}: {count} rows")

    # Overall timing summary
    overall_secs = time.perf_counter() - overall_start
    completed_topics = len(topic_times)
    if completed_topics:
        avg_secs = sum(topic_times) / completed_topics
        print(f"Processed {completed_topics} topics in {overall_secs:.1f}s (avg {avg_secs:.1f}s/topic)")


if __name__ == "__main__":
    main()


