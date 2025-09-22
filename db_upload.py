# db_upload.py
# Usage:
#   python db_upload.py \
#     --in questions_hf_explained.json \
#     --out-json questions_hf_explained.json \
#     --out-csv questions_hf_csv.csv
#
#   # If you ALSO want to replace every topic_id first:
#   python db_upload.py \
#     --in questions_asthma_explained.json \
#     --new-topic-id "a8769dfc-2521-49be-99c7-f158816dce46" \
#     --out-json questions_asthma_explained.json \
#     --out-csv questions_asthma_csv.csv
#
# Notes:
# - If --new-topic-id is omitted, topic_id values are left as-is.
# - Writes an updated JSON (preserving all extra fields).
# - Exports a CSV matching your Supabase 'questions' table columns + correct_answer index:
#   topic_id, type, stem, is_active, explanation_eli5, explanation_l1_points, explanation_l2, options, correct_answer, created_at, updated_at
# - Assumes 'options' and 'explanation_l1_points' are JSON/JSONB columns in Supabase.
# - CSV 'correct_answer' is an INTEGER 0–4. Script accepts either a numeric index (0–4) **or** a letter ("A"–"E") in the input JSON.

import argparse
import csv
import json
from datetime import datetime, timezone
from typing import List, Dict, Any


CSV_COLUMNS = [
    "topic_id",
    "type",
    "stem",
    "is_active",
    "explanation_eli5",
    "explanation_l1_points",
    "explanation_points_by_option",
    "explanation_l2",
    "options",
    "correct_answer",   # integer index 0-4
    "created_at",
    "updated_at",
]


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of question objects.")
    return data


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iso_now() -> str:
    # ISO 8601 with Zulu suffix, e.g. 2025-08-22T13:45:12Z
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_correct_index(val: Any) -> int:
    """
    Accepts either:
      - int 0..4
      - numeric string "0".."4"
      - letter "A".."E" (case-insensitive)
    Returns -1 if invalid.
    """
    if isinstance(val, int):
        return val if 0 <= val <= 4 else -1
    s = str(val).strip()
    if s.isdigit():
        i = int(s)
        return i if 0 <= i <= 4 else -1
    s = s.upper()
    pos = "ABCDE".find(s)
    return pos if pos != -1 else -1


def normalise_question(
    q: Dict[str, Any],
    topic_id: str,
    default_is_active: bool,
    created_at: str,
    updated_at: str,
) -> Dict[str, Any]:
    """
    Build a row for CSV export. Does NOT modify the original question dict.
    """
    stem = q.get("stem", "")
    options = q.get("options", [])
    if not isinstance(options, list):
        options = []

    # Explanations
    explanation_eli5 = q.get("explanation_eli5", "")
    explanation_l1_points = q.get("explanation_l1_points", [])
    points_by_option = q.get("explanation_points_by_option", {})
    explanation_l2 = q.get("explanation_l2", "")

    if not isinstance(explanation_l1_points, list):
        # If not provided but we have points_by_option and a correct index, 
        # derive l1 from the correct option mapping for backward compatibility.
        if isinstance(points_by_option, dict):
            ci = get_correct_index(q.get("correct_answer", ""))
            key = str(ci)
            if key in points_by_option and isinstance(points_by_option[key], list):
                explanation_l1_points = points_by_option[key]
            else:
                explanation_l1_points = [str(explanation_l1_points)]
        else:
            explanation_l1_points = [str(explanation_l1_points)]

    # Correct answer index (0–4) from input (supports int or letter)
    idx = get_correct_index(q.get("correct_answer", ""))

    # Basic sanity checks / warnings
    if len(options) != 5:
        print(f"Warning: options length != 5 for stem: {stem[:80]}...")
    if idx < 0 or idx > 4:
        print(
            f"Warning: invalid correct_answer for stem: {stem[:80]}... -> {q.get('correct_answer')}")

    row = {
        "topic_id": topic_id,
        "type": q.get("type", "MCQ"),
        "stem": stem,
        "is_active": bool(q.get("is_active", default_is_active)),
        "explanation_eli5": str(explanation_eli5),
        # JSON-encode arrays/objects for CSV -> Postgres JSONB
        "explanation_l1_points": json.dumps(explanation_l1_points, ensure_ascii=False),
        "explanation_points_by_option": json.dumps(points_by_option if isinstance(points_by_option, dict) else {}, ensure_ascii=False),
        "explanation_l2": str(explanation_l2),
        "options": json.dumps(options, ensure_ascii=False),
        "correct_answer": idx,  # integer 0–4
        "created_at": created_at,
        "updated_at": updated_at,
    }
    return row


def main():
    ap = argparse.ArgumentParser(
        description="Optionally replace topic_id in question JSON and export a Supabase-ready CSV (with correct_answer index)."
    )
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Input questions JSON file.")
    ap.add_argument("--new-topic-id", dest="new_topic_id", default="",
                    help="Optional new topic_id (UUID from Supabase).")
    ap.add_argument("--out-json", dest="out_json", required=True,
                    help="Path to write updated JSON.")
    ap.add_argument("--out-csv", dest="out_csv",
                    required=True, help="Path to write CSV.")
    ap.add_argument(
        "--is-active",
        dest="is_active",
        default="true",
        help="Default is_active value for all rows (true/false). Default: true",
    )
    ap.add_argument(
        "--created-at",
        dest="created_at",
        default="",
        help="Optional ISO timestamp for created_at (e.g., 2025-08-22T12:00:00Z). Defaults to now.",
    )
    ap.add_argument(
        "--updated-at",
        dest="updated_at",
        default="",
        help="Optional ISO timestamp for updated_at. Defaults to now.",
    )
    args = ap.parse_args()

    questions = load_json(args.in_path)

    # Resolve timestamps
    created_at = args.created_at.strip() or iso_now()
    updated_at = args.updated_at.strip() or iso_now()
    default_is_active = str2bool(args.is_active)

    # 1) Optionally update topic_id across all questions
    if args.new_topic_id.strip():
        for q in questions:
            q["topic_id"] = args.new_topic_id.strip()

    # 2) Save updated JSON (preserve all fields exactly)
    save_json(args.out_json, questions)

    # 3) Build CSV rows (using per-row topic_id: replaced if provided, else original)
    rows_for_csv: List[Dict[str, Any]] = []
    for q in questions:
        topic_id_for_row = q.get("topic_id", args.new_topic_id.strip())
        row = normalise_question(
            q=q,
            topic_id=topic_id_for_row,
            default_is_active=default_is_active,
            created_at=created_at,
            updated_at=updated_at,
        )
        rows_for_csv.append(row)

    # 4) Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows_for_csv:
            writer.writerow(r)

    print(f"Updated JSON -> {args.out_json}")
    print(f"CSV written   -> {args.out_csv}")
    print("CSV columns   -> " + ", ".join(CSV_COLUMNS))


if __name__ == "__main__":
    main()
