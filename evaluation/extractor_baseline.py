"""
extractor_baseline.py

Deliberately naive counterpart to extractor.py: it does NOT search the model
output for a number anywhere, apply any of extractor.py's heuristics, or look
at the prompt/expression at all. It only checks whether the very first thing
the model generates -- before any space, punctuation, letter, etc. -- is a
number, and whether that number equals the gold answer.

This exists as a baseline lower bound: how much of the accuracy measured by
the full extractor.py actually comes from the model stating the answer
immediately, versus extractor.py's more lenient "find the answer somewhere in
the output" heuristics.

Output schema matches extractor.py's (extracted_answer, contains_gold_label,
extractor, correct) so results can be fed straight into the same
analysis/behavioral/summarize_results.py used for the main extractor.

Usage
-----
python extractor_baseline.py --in-dir results/raw/base_and_instruct \\
                              --out-dir results/raw/base_and_instruct_baseline
"""

import argparse
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional


def normalize_unicode_digits(text: str) -> str:
    out = []
    for ch in text:
        if ch.isdigit() and ch not in "0123456789":
            try:
                out.append(str(unicodedata.digit(ch)))
            except (TypeError, ValueError):
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


# Sign + digits, with at most one decimal separator ("." or ",").
# Stops at the first character that isn't part of the number (space,
# punctuation, letter, ...), which is exactly what re.match does by nature.
LEADING_NUMBER_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?")


def extract_leading_number(model_output_raw: str) -> Optional[str]:
    """The number at the very start of the model's output, or None if the
    output doesn't start with one (after skipping only leading whitespace)."""
    if not model_output_raw:
        return None
    text = normalize_unicode_digits(model_output_raw).lstrip()
    m = LEADING_NUMBER_RE.match(text)
    return m.group(0) if m else None


def _to_decimal(token: str) -> Optional[Decimal]:
    try:
        return Decimal(str(token).strip().replace(",", "."))
    except (InvalidOperation, ValueError, TypeError):
        return None


def matches_gold(extracted: Optional[str], gold_answer) -> bool:
    if extracted is None or gold_answer is None:
        return False
    gold_s = normalize_unicode_digits(str(gold_answer).strip())
    if extracted == gold_s:
        return True
    a, b = _to_decimal(extracted), _to_decimal(gold_s)
    return a is not None and b is not None and a == b


def iter_jsonl_files(base_dir: Path):
    for jsonl in sorted(base_dir.rglob("*.jsonl")):
        yield jsonl


def _normalize_model_name(s: str) -> str:
    s = s.strip().strip("/").split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _model_matches_filter(model_folder: str, filters: set) -> bool:
    if not filters:
        return True
    folder_norm = _normalize_model_name(model_folder)
    for f in filters:
        fn = _normalize_model_name(f)
        if fn and (fn == folder_norm or fn in folder_norm or folder_norm in fn):
            return True
    return False


def process_file(in_path: Path, in_dir: Path, out_dir: Path, model_filters: set) -> int:
    # Mirrors run_eval.py's out_dir/<model_slug>/<lang>/<variant>.jsonl layout.
    model_folder = in_path.parent.parent.name
    if not _model_matches_filter(model_folder, model_filters):
        return 0

    out_path = out_dir / in_path.relative_to(in_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            leading = extract_leading_number(rec.get("model_output_raw", ""))
            rec["extracted_answer"] = [leading] if leading is not None else []
            rec["contains_gold_label"] = matches_gold(leading, rec.get("answer"))
            rec["extractor"] = "leading_number_baseline" if leading is not None else "unknown"
            # Actual accuracy is derived downstream from contains_gold_label
            # (see summarize_results.py); kept here only for schema parity.
            rec["correct"] = False

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows += 1

    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Baseline extractor: accepts a number only if it is the very "
                     "first thing the model generates, before any space/punctuation."
    )
    ap.add_argument("--in-dir", required=True, help="Input results root (raw generation JSONLs).")
    ap.add_argument("--out-dir", required=True, help="Output root (new JSONLs; does not overwrite input).")
    ap.add_argument("--models", default="", help="Optional comma-separated model folder name filters.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_filters = {m.strip() for m in args.models.split(",") if m.strip()}

    processed_files = 0
    processed_rows = 0
    for in_path in iter_jsonl_files(in_dir):
        rows = process_file(in_path, in_dir, out_dir, model_filters)
        if rows:
            processed_files += 1
            processed_rows += rows

    print(f"Done. Processed files={processed_files}, rows={processed_rows}")
    print(f"Output written to: {out_dir}")
    if model_filters and processed_files == 0:
        print(
            "No files matched --models filter. "
            "Filters are matched against model folder names using case-insensitive normalized substring matching."
        )


if __name__ == "__main__":
    main()
