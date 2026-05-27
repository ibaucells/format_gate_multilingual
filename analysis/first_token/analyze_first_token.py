#!/usr/bin/env python3
"""
analyze_first_token.py

Analyzes one or more first-token CSV files produced by first_token_table.py.

For each file, prints:

  1. First-token confusion table
      Rows = top N first tokens (by overall frequency).
      Columns = outcome (answer / no_answer).
      Answers: for each token, how does it split across outcomes?

  2. Outcome × first-token matrix
      Rows = outcome (no_answer / answer).
     Columns = top N first tokens (by overall frequency) + OTHER.
     Answers: of all items with outcome X, how many used token T first?

  3. No-answer first-token breakdown by lang × prompt
     For each (lang, prompt), shows the top first tokens among no_answer items
     with their counts and share of that (lang, prompt)'s no_answer total.

Usage:
  python analyze_first_token.py first_token_analysis/first_token_table_salamandra-7b.csv
  python analyze_first_token.py first_token_analysis/*.csv
  python analyze_first_token.py first_token_analysis/*.csv --lang ar --prompt answer
  python analyze_first_token.py first_token_analysis/*.csv --top-tokens 15
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "  n/a"
    return f"{100 * n / total:5.1f}%"


def _bar(n: int, total: int, width: int = 15) -> str:
    if total == 0:
        return " " * width
    filled = round(width * n / total)
    return "█" * filled + "░" * (width - filled)


def load_rows(path: Path,
              lang_filter: Optional[str],
              prompt_filter: Optional[str],
              type_filter: Optional[str]):
    rows = []
    normalized_prompt_filter = normalize_prompt(prompt_filter) if prompt_filter else None
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["prompt"] = normalize_prompt(row.get("prompt", ""))
            if lang_filter   and row["lang"]   != lang_filter:
                continue
            if normalized_prompt_filter and row["prompt"] != normalized_prompt_filter:
                continue
            if type_filter   and row["type"]   != type_filter:
                continue
            rows.append(row)
    return rows


def outcome(row: dict) -> str:
    if int(row["no_answer"]):
        return "no_answer"
    return "answer"


def tok_label(row: dict) -> str:
    """Decoded token string if present, otherwise the raw integer id."""
    decoded = row.get("first_token_decoded", "")
    return decoded if decoded else row.get("first_output_id", "?")


_PROMPT_COLLAPSE = {"answer_correct": "answer", "answer_incorrect": "answer"}

def normalize_prompt(prompt: str) -> str:
    return _PROMPT_COLLAPSE.get(prompt, prompt)


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def analyze_file(path: Path,
                 lang_filter: Optional[str],
                 prompt_filter: Optional[str],
                 type_filter: Optional[str],
                 top_tokens: int):

    print(f"\n{'=' * 78}")
    print(f"FILE: {path}")
    print('=' * 78)

    rows = load_rows(path, lang_filter, prompt_filter, type_filter)
    if not rows:
        print("  (no rows match the current filters)")
        return

    # ── pre-group rows by token and by outcome ─────────────────────────────
    tok_rows   = defaultdict(list)   # token  → [row, ...]
    out_rows   = defaultdict(list)   # outcome→ [row, ...]
    for row in rows:
        tl = tok_label(row)
        ol = outcome(row)
        tok_rows[tl].append(row)
        out_rows[ol].append(row)

    # Top-N tokens by overall frequency
    top = sorted(tok_rows, key=lambda k: -len(tok_rows[k]))[:top_tokens]
    top_set = set(top)

    tok_w = max(len("first_token"), max((len(t) for t in top), default=0)) + 1
    total  = len(rows)

    # ── 1. First-token confusion table ─────────────────────────────────────
    # Rows = tokens, Columns = answer / no_answer
    print(f"\n--- 1. First-token confusion table  (top {top_tokens} by frequency) ---\n")
    print(f"  Rows: first token  |  Columns: outcome  (% = share of that token's total)\n")

    hdr1 = (f"  {'first_token':<{tok_w}}"
            f" | {'total':>6}"
            f" | {'answer':>7}       "
            f" | {'no_answer':>9}       "
            f" | ans%")
    print(hdr1)
    print("  " + "-" * (len(hdr1) - 2))
    for tok in top:
        r = tok_rows[tok]
        n_tot = len(r)
        n_ans = sum(1 for x in r if outcome(x) == "answer")
        n_na  = sum(1 for x in r if outcome(x) == "no_answer")
        bar   = _bar(n_ans, n_tot)
        print(f"  {tok:<{tok_w}}"
              f" | {n_tot:>6}"
              f" | {n_ans:>7} ({fmt_pct(n_ans,  n_tot)})"
              f" | {n_na:>9} ({fmt_pct(n_na, n_tot)})"
              f" | {fmt_pct(n_ans, n_tot)}  {bar}")

    # ── 2. Outcome × first-token matrix ────────────────────────────────────
    # Rows = outcome, Columns = top tokens + OTHER
    # Cell value = count + % of that outcome's total
    print(f"\n--- 2. Outcome × first-token matrix  (top {top_tokens} tokens + OTHER) ---\n")
    print(f"  Rows: outcome  |  Columns: first token  (% = share of that outcome's total)\n")

    OUTCOMES = [("no_answer", "no_answer"),
                ("answer",    "answer")]

    # Column header: each token truncated to col_w chars
    col_w = max(7, max((len(t) for t in top), default=7))
    col_w = min(col_w, 14)   # cap display width

    # header row
    outcome_label_w = max(len(lbl) for _, lbl in OUTCOMES) + 2
    header_cells = [f"{t[:col_w]:^{col_w+3}}" for t in top] + [f"{'OTHER':^{col_w+3}}"] + [f"{'TOTAL':>7}"]
    print("  " + " " * outcome_label_w + "  " + " ".join(header_cells))
    print("  " + "-" * (outcome_label_w + 2 + (col_w + 4) * (len(top) + 1) + 8))

    for out_key, out_label in OUTCOMES:
        r_out = out_rows[out_key]
        n_out = len(r_out)

        cells = []
        for tok in top:
            n_cell = sum(1 for x in r_out if tok_label(x) == tok)
            cells.append(f"{n_cell:>{col_w}} ({fmt_pct(n_cell, n_out)})")

        # OTHER = items in this outcome whose token is NOT in top
        n_other = sum(1 for x in r_out if tok_label(x) not in top_set)
        cells.append(f"{n_other:>{col_w}} ({fmt_pct(n_other, n_out)})")

        row_str = "  ".join(cells)
        print(f"  {out_label:<{outcome_label_w}}  {row_str}  {n_out:>6}")

    # ── 3. No-answer first-token breakdown by lang × prompt ────────────────
    print("\n--- 3. No-answer first tokens by lang × prompt ---\n")
    print("  For each (lang, prompt): top first tokens among no_answer items.\n")

    # group no_answer rows by (lang, prompt)
    lp_na = defaultdict(list)
    for row in out_rows["no_answer"]:
        lp_na[(row["lang"], normalize_prompt(row["prompt"]))].append(row)

    if not lp_na:
        print("  (no no_answer items found)")
        return

    lang_w   = max(len("lang"),   max(len(k[0]) for k in lp_na)) + 1
    prompt_w = max(len("prompt"), max(len(k[1]) for k in lp_na)) + 1

    hdr3 = (f"  {'lang':<{lang_w}} | {'prompt':<{prompt_w}}"
            f" | {'no_ans_total':>12}"
            f" | {'first_token':<{tok_w}}"
            f" | {'count':>5} | {'% of no_ans':>11}")
    print(hdr3)
    print("  " + "-" * (len(hdr3) - 2))

    for key in sorted(lp_na):
        lang, prompt = key
        na_rows = lp_na[key]
        na_total = len(na_rows)

        # count tokens within this (lang, prompt) no_answer group
        tok_cnt = defaultdict(int)
        for row in na_rows:
            tok_cnt[tok_label(row)] += 1

        sorted_toks = sorted(tok_cnt, key=lambda k: -tok_cnt[k])
        first = True
        for tok in sorted_toks:
            cnt = tok_cnt[tok]
            if first:
                lang_str   = f"{lang:<{lang_w}}"
                prompt_str = f"{prompt:<{prompt_w}}"
                total_str  = f"{na_total:>12}"
                first = False
            else:
                lang_str   = " " * lang_w
                prompt_str = " " * prompt_w
                total_str  = " " * 12
            print(f"  {lang_str} | {prompt_str} | {total_str}"
                  f" | {tok:<{tok_w}}"
                  f" | {cnt:>5} | {fmt_pct(cnt, na_total):>11}  {_bar(cnt, na_total)}")
        print()

    # ── 4. No-answer first-token matrix by lang (aggregated over prompts) ──
    print(f"\n--- 4. No-answer first-token matrix by lang  (top {top_tokens} tokens + OTHER) ---\n")
    print("  Rows: language  |  Columns: first token among no_answer items  (% = share of that lang's no_answer total)\n")

    # group no_answer rows by lang only
    lang_na = defaultdict(list)
    for row in out_rows["no_answer"]:
        lang_na[row["lang"]].append(row)

    # top tokens within no_answer subset specifically
    na_tok_cnt = defaultdict(int)
    for row in out_rows["no_answer"]:
        na_tok_cnt[tok_label(row)] += 1
    top_na     = sorted(na_tok_cnt, key=lambda k: -na_tok_cnt[k])[:top_tokens]
    top_na_set = set(top_na)

    lang_w4 = max(len("lang"), max((len(lg) for lg in lang_na), default=4)) + 1

    # column header
    col_w4 = max(7, max((len(t) for t in top_na), default=7))
    col_w4 = min(col_w4, 14)
    header_cells4 = [f"{t[:col_w4]:^{col_w4+3}}" for t in top_na] + [f"{'OTHER':^{col_w4+3}}"] + [f"{'TOTAL':>7}"]
    print("  " + " " * (lang_w4 + 2) + "  " + " ".join(header_cells4))
    print("  " + "-" * (lang_w4 + 2 + (col_w4 + 4) * (len(top_na) + 1) + 8))

    for lang in sorted(lang_na):
        r_lang = lang_na[lang]
        n_lang = len(r_lang)

        cells4 = []
        for tok in top_na:
            n_cell = sum(1 for x in r_lang if tok_label(x) == tok)
            cells4.append(f"{n_cell:>{col_w4}} ({fmt_pct(n_cell, n_lang)})")

        n_other4 = sum(1 for x in r_lang if tok_label(x) not in top_na_set)
        cells4.append(f"{n_other4:>{col_w4}} ({fmt_pct(n_other4, n_lang)})")

        row_str4 = "  ".join(cells4)
        print(f"  {lang:<{lang_w4}}  {row_str4}  {n_lang:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Analyze first-token CSV files produced by first_token_table.py."
    )
    ap.add_argument("files", nargs="+", help="CSV file(s) to analyze.")
    ap.add_argument("--lang",   default=None, help="Filter to a single language code.")
    ap.add_argument("--prompt", default=None, help="Filter to a single prompt value.")
    ap.add_argument("--type",   default=None, help="Filter to a single problem type.")
    ap.add_argument(
        "--top-tokens", type=int, default=20,
        help="Number of top first-tokens to show (default: 20).",
    )
    args = ap.parse_args()

    found = False
    for pattern in args.files:
        paths = sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        for path in paths:
            if not path.exists():
                print(f"Warning: {path} not found — skipping.", file=sys.stderr)
                continue
            found = True
            analyze_file(
                path,
                lang_filter=args.lang,
                prompt_filter=args.prompt,
                type_filter=args.type,
                top_tokens=args.top_tokens,
            )

    if not found:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
