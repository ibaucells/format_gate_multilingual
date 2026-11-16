import argparse
import os
import json
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Optional, Tuple


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
        if not fn:
            continue
        if fn == folder_norm or fn in folder_norm or folder_norm in fn:
            return True
    return False


def _model_name_leaf(model_value) -> str:
    if model_value is None:
        return ""
    return str(model_value).strip().strip("/").split("/")[-1]


def _record_key_from_input_record(rec: dict):
    return (
        str(rec.get("id", "")),
        _model_name_leaf(rec.get("model")),
        str(rec.get("language", "")),
        str(rec.get("prompt_variant", "")),
        str(rec.get("type", "")),
        str(rec.get("subtype", "")),
        str(rec.get("expression", "")),
    )


def _record_key_from_sampled_row(row: dict):
    return (
        str(row.get("id", "")),
        _model_name_leaf(row.get("model")),
        str(row.get("lang", row.get("language", ""))),
        str(row.get("prompt", row.get("prompt_variant", ""))),
        str(row.get("type", "")),
        str(row.get("subtype", "")),
        str(row.get("expression", "")),
    )


def _load_sample_keys_from_jsonl(sample_jsonl: Path):
    sampled_keys = set()
    with sample_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sampled_keys.add(_record_key_from_sampled_row(row))
    return sampled_keys


def _ended_with_eos(rec: dict) -> Optional[bool]:
    output_ids = rec.get("output_ids")
    if not isinstance(output_ids, list) or not output_ids:
        return None
    last_id = output_ids[-1]

    gen = rec.get("generation_kwargs") or {}
    eos = gen.get("eos_token_id")
    if eos is None:
        return None
    if isinstance(eos, list):
        return last_id in eos
    return last_id == eos


def _map_norm_span_to_raw(raw_text: str, norm_text: str, span: Optional[Tuple[int, int]]):
    if span is None:
        return None
    a, b = span
    val = norm_text[a:b]
    if raw_text == norm_text:
        return (a, b)
    m = re.search(rf"(?<!\d){re.escape(val)}(?!\d)", raw_text)
    if m:
        return m.span(0)
    return None


def _parse_decimal_number_token(token: str) -> Optional[Decimal]:
    t = normalize_unicode_digits((token or "").strip())
    if not re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", t):
        return None
    t = t.replace(",", ".")
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def _find_gold_label_token_in_output(answer_value, model_output_raw: str) -> Optional[str]:
    if answer_value is None:
        return None
    ans_s = normalize_unicode_digits(str(answer_value).strip())
    if not ans_s:
        return None

    ans_dec = _parse_decimal_number_token(ans_s)
    text = normalize_unicode_digits(model_output_raw or "")
    num_pat = re.compile(r"(?<![\d.,_])([+-]?\d+(?:[.,]\d+)?)(?![\d_])")
    for m in num_pat.finditer(text):
        tok = m.group(1)
        if tok == ans_s:
            return tok
        if ans_dec is not None:
            tok_dec = _parse_decimal_number_token(tok)
            if tok_dec is not None and tok_dec == ans_dec:
                return tok
    # Also match spaced-sign numbers like "- 12" which normalize to "-12".
    # Use (?<!\d ) to avoid matching binary operators as in "5 - 12".
    if ans_dec is not None:
        spaced_pat = re.compile(r"(?<!\d )([+-])\s+(\d+(?:[.,]\d+)?)(?![\d_])")
        for m in spaced_pat.finditer(text):
            compact = m.group(1) + m.group(2)
            tok_dec = _parse_decimal_number_token(compact)
            if tok_dec is not None and tok_dec == ans_dec:
                return compact
    return None


_SPURIOUS_CHECK_EXTRACTORS = {
    "unresolved_operation_loop",
}


def _is_gold_label_spurious(gold_answer: str, model_output_raw: str) -> bool:
    """
    Returns True if ALL occurrences of gold_answer in model_output_raw appear only
    as an operand inside an unanswered arithmetic expression (e.g. "100 - 92 ="),
    meaning the gold label is present purely by coincidence, not as an actual answer.

    Operand position: the rest of the line after the gold consists only of arithmetic
    characters followed by an unanswered "=", and the gold is not immediately after "=".
    Truncated last lines (cut off before "=") are also accepted as operand position if
    the gold is immediately preceded by an arithmetic operator.
    """
    # Normalize so Eastern Arabic/Indic digits match the same way as
    # _find_gold_label_token_in_output does.
    text = normalize_unicode_digits(model_output_raw)
    gold = normalize_unicode_digits(gold_answer).strip()

    # Use digit boundaries so "25" doesn't match inside "125" or "25600".
    escaped = rf"(?<!\d){re.escape(gold)}(?!\d)"
    occurrences = list(re.finditer(escaped, text))
    if not occurrences:
        return False

    for m in occurrences:
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        if line_end < 0:
            line_end = len(text)
        before = text[line_start:m.start()]
        after = text[m.end():line_end]
        # Handle truncated last line: generation cut off before the "=".
        is_truncated_last_line = (line_end == len(text) and not text.rstrip().endswith("="))
        if is_truncated_last_line:
            # Require that before ends with an operator or "(", confirming operand position.
            # Include "x" and ":" as mul/div operators used in some languages.
            preceded_by_op = bool(re.search(r"[+\-*/×÷(:x]\s*$", before))
            after_ok = preceded_by_op and bool(re.fullmatch(r"[\d\s()\[\]+\-*/×÷.:x,]*", after))
        else:
            # Normal case: after the gold must be arithmetic chars then an unanswered "=".
            after_ok = bool(re.fullmatch(r"[\d\s()\[\]+\-*/×÷.:x,]*=\s*", after))
        # Gold must not be in answer position (i.e. immediately after "=").
        not_answer_pos = not bool(re.search(r"=\s*$", before))
        if not (after_ok and not_answer_pos):
            return False
    return True


def _normalize_extracted_number_token(token: str) -> str:
    tok = normalize_unicode_digits(str(token or ""))
    tok = re.sub(r"^([+-])\s+(?=\d|[.,]\d)", r"\1", tok)
    tok = re.sub(r"\s*/\s*", "/", tok)
    return tok


def _process_file(task):
    (
        in_path_str,
        in_dir_str,
        out_dir_str,
        model_filters,
        sampled_keys,
        only_sampled_rows,
        single_jsonl_mode,
    ) = task
    in_path = Path(in_path_str)
    in_dir = Path(in_dir_str)
    out_dir = Path(out_dir_str)

    model_folder = in_path.parent.parent.name
    if not _model_matches_filter(model_folder, model_filters):
        return False, 0

    out_path = None if single_jsonl_mode else out_dir / in_path.relative_to(in_dir)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_any = False
    processed_rows = 0
    output_lines = []

    fout_cm = (
        out_path.open("w", encoding="utf-8")
        if out_path is not None
        else open(os.devnull, "w", encoding="utf-8")
    )
    with in_path.open("r", encoding="utf-8") as fin, fout_cm as fout:
        for line in fin:
            rec = json.loads(line)
            apply_this_record = True
            if sampled_keys is not None:
                apply_this_record = _record_key_from_input_record(rec) in sampled_keys

            if not apply_this_record:
                if not only_sampled_rows:
                    rec["extractor"] = "unknown"
                    rec["correct"] = False
                    rec.pop("needs_llm", None)
                    rec.setdefault("correct", False)
                    serialized = json.dumps(rec, ensure_ascii=False) + "\n"
                    if out_path is None:
                        output_lines.append(serialized)
                    else:
                        fout.write(serialized)
                    wrote_any = True
                    processed_rows += 1
                continue

            gold_token = _find_gold_label_token_in_output(
                rec.get("answer"),
                rec.get("model_output_raw", ""),
            )
            rec["contains_gold_label"] = gold_token is not None
            extracted_list, extractor, _needs_llm = extract_high_confidence_only(
                rec.get("prompt_text", ""),
                rec.get("model_output_raw", ""),
                expression=rec.get("expression", ""),
                prompt_variant=rec.get("prompt_variant", ""),
                problem_type=rec.get("type", ""),
                generation_ended_with_eos=_ended_with_eos(rec),
            )
            rec["extracted_answer"] = [_normalize_extracted_number_token(x) for x in extracted_list]
            rec["extractor"] = extractor if extractor else "unknown"
            rec["correct"] = False

            if rec["contains_gold_label"] and rec["extractor"] in _SPURIOUS_CHECK_EXTRACTORS:
                if _is_gold_label_spurious(rec.get("answer", ""), rec.get("model_output_raw", "")):
                    rec["contains_gold_label"] = "yes_but_spurious"

            rec.pop("needs_llm", None)
            rec.setdefault("correct", False)

            serialized = json.dumps(rec, ensure_ascii=False) + "\n"
            if out_path is None:
                output_lines.append(serialized)
            else:
                fout.write(serialized)
            wrote_any = True
            processed_rows += 1

    if out_path is None:
        return wrote_any, processed_rows, output_lines

    if not wrote_any:
        out_path.unlink(missing_ok=True)

    return wrote_any, processed_rows, None


def _leading_standalone_number_line(text: str, number_re: str, prompt_text: str = ""):
    prefix_sym_re = r"(?:[$€£¥₹])"
    spaced_sign_number_re = (
        r"(?:[+-]\s+(?:\d+(?:[.,]\d+)?|\d{1,3}(?:[.,_ ]\d{3})+(?:[.,]\d+)?|[.,]\d+))"
    )
    prompt_nums = set(re.findall(number_re, normalize_unicode_digits(prompt_text or "")))
    for mline in re.finditer(r"(?m)^.*$", text):
        raw = mline.group(0)
        if not raw.strip():
            continue
        # Skip leading punctuation-only lines (e.g., “．”, “?”, “...”)
        if re.fullmatch(r"\s*[.。．，,;；:：!?！？’\”””’’`~\-_=+*/×÷|\\()\[\]{}<>《》【】]+\s*", raw):
            continue
        lead_punct_re = r'[.。．，,;；:：!?！？\'"~_|\\()\[\]{}]'
        m = re.fullmatch(
            rf"\s*(?:{lead_punct_re}+\s*)?(?:{prefix_sym_re}\s*)?({number_re})\s*(?:[%٪]|[.،,:;!?。]|\\\]|\\\)|\$)?\s*",
            raw,
        )
        if m:
            abs_start = mline.start() + m.start(1)
            abs_end = mline.start() + m.end(1)
            return (abs_start, abs_end)
        spaced_sign = re.fullmatch(
            rf"\s*({spaced_sign_number_re})\s*(?:[%٪]|[.،,:;!?。]|\\\]|\\\)|\$)?\s*",
            raw,
        )
        if spaced_sign:
            abs_start = mline.start() + spaced_sign.start(1)
            abs_end = mline.start() + spaced_sign.end(1)
            return (abs_start, abs_end)
        # CJK confirmation line: number + optional unit + CJK explanation ending in "。"
        # e.g. "800 (元) 答:每人需付 800 元。" — the echoed 800 is just confirmation.
        cjk_m = re.match(
            rf"\s*(?:{prefix_sym_re}\s*)?({number_re})\s*(?:\([^)\n]{{1,16}}\))?\s*[\u4E00-\u9FFF]",
            raw,
        )
        if cjk_m and raw.rstrip().endswith("\u3002"):
            captured = cjk_m.group(1)
            tail_nums = re.findall(number_re, raw[cjk_m.end(1):])
            if all(n == captured or n in prompt_nums for n in tail_nums):
                return (mline.start() + cjk_m.start(1), mline.start() + cjk_m.end(1))
        break
    return None


def _leading_prefix_number(text: str):
    # First non-empty line starts with a number, followed by punctuation or
    # obvious equation-like continuation; keep the leading number.
    simple_lead_num_re = r"[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)"
    for mline in re.finditer(r"(?m)^.*$", text):
        raw = mline.group(0)
        s = raw.strip()
        if not s:
            continue
        m = re.match(rf"^\s*({simple_lead_num_re})", raw)
        if not m:
            return None
        after = raw[m.end() :]

        if re.match(rf"\s+{simple_lead_num_re}(?![\d.,])", after):
            return (mline.start() + m.start(1), mline.start() + m.end(1))
        if re.match(rf"\s*[，,;；:：]\s*{simple_lead_num_re}\s*[+\-*/^×÷:：]", after):
            return (mline.start() + m.start(1), mline.start() + m.end(1))
        if re.match(rf"\s*[.。]\s*{simple_lead_num_re}\s*[.。]", after):
            return (mline.start() + m.start(1), mline.start() + m.end(1))
        if re.match(rf"\s*[Rr]\s*{simple_lead_num_re}(?![\d.,])", after):
            return (mline.start() + m.start(1), mline.start() + m.end(1))
        if re.match(r"\s*(?:[;；,:.。])", after):
            return (mline.start() + m.start(1), mline.start() + m.end(1))
        if "=" in raw or "≈" in raw:
            return None
        if re.match(r"\s*(?:[;；,:.。]|$)", after) or ("=" in after and re.search(r"[+\-*/^×÷]", after)):
            return (mline.start() + m.start(1), mline.start() + m.end(1))
        return None
    return None


def _leading_number_with_unit_explanation(text: str, number_re: str, prompt_text: str = ""):
    # First non-empty line like: "84 (cm)" or "78 km - ..."
    prefix_sym_re = r"(?:[$€£¥₹])"
    prompt_nums = set(re.findall(number_re, normalize_unicode_digits(prompt_text or "")))
    tail_text_re = re.compile(
        r"^[\s\.,;:!?%٪()\[\]{}'\"“”‘’\-–—/\\"
        r"A-Za-z\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF]*$"
    )
    for mline in re.finditer(r"(?m)^.*$", text):
        raw = mline.group(0)
        if not raw.strip():
            continue
        unit_pat = (
            r"^\s*(?:" + prefix_sym_re + r"\s*)?(" + number_re + r")\s*"
            r"(?:\([^)\n]{1,16}\)|\\text\{[^}\n]{1,32}\}|[^\W\d_]{1,8})"
        )
        m = re.match(unit_pat, raw)
        if m:
            captured_num = m.group(1)
            tail = raw[m.end():]
            # Treat a numbered-list sequence as a sentence/line boundary and
            # crop the tail there (e.g. "783 厘米 2. 1000… 3. 1000…" → tail
            # cropped to "").  Only apply when at least two *sequential*
            # list-item numbers exist in the tail (e.g. "2." followed by "3.")
            # so lone answer values like "The answer is 10." are not affected.
            _li_pat = re.compile(r"\s+(\d{1,3})\.\s")
            _li_matches = list(_li_pat.finditer(tail))
            if len(_li_matches) >= 2:
                _li_nums = [int(_lm.group(1)) for _lm in _li_matches]
                _has_seq = any(
                    _li_nums[i + 1] == _li_nums[i] + 1
                    for i in range(len(_li_nums) - 1)
                )
                if _has_seq:
                    tail = tail[: _li_matches[0].start()]
            # Reject arithmetic operators.  Allow a standalone dash used as a
            # text separator (e.g. "50 (км/ч) - скорость...") by only matching
            # '-' when it is adjacent to a digit (i.e. truly arithmetic).
            if re.search(r"[+*/=×÷]|\d\s*-|-\s*\d", tail):
                # Fallback: accept when the leading number is not a prompt
                # number and the tail continues with more computation
                # (space + digit), meaning the model stated the answer first
                # then showed its work (e.g. "240 (平方厘米) 240 / 2 = 120…").
                if captured_num not in prompt_nums and re.match(r"\s+\d", tail):
                    return (mline.start() + m.start(1), mline.start() + m.end(1))
                return None
            tail_nums = list(re.finditer(number_re, tail))
            # Allow the captured answer to echo in the tail
            # (e.g. "200 (個) 答: 200" — the repeated 200 is a confirmation).
            if any(
                tm.group(0) not in prompt_nums and tm.group(0) != captured_num
                for tm in tail_nums
            ):
                return None
            tail_wo_nums = tail
            for tm in tail_nums:
                tail_wo_nums = re.sub(re.escape(tm.group(0)), " ", tail_wo_nums, count=1)
            if not tail_text_re.fullmatch(tail_wo_nums):
                return None
            return (mline.start() + m.start(1), mline.start() + m.end(1))
        return None
    return None


def _leading_number_with_explanation(text: str, number_re: str):
    # First non-empty line starts with one number and then explanatory text
    # (e.g., "6 \\) (not a perfect square)" or "- 305 (rounded)").
    # Avoid equations/continued math.
    letter_re = r"[A-Za-z\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF]"
    unsigned_number_re = r"(?:\d+(?:[.,]\d+)?|\d{1,3}(?:[.,_ ]\d{3})+(?:[.,]\d+)?|[.,]\d+)"
    prefix_sym_re = r"(?:[$€£¥₹])"
    for mline in re.finditer(r"(?m)^.*$", text):
        raw = mline.group(0)
        if not raw.strip():
            continue
        m = re.match(
            rf"^\s*(?:(?:[*•]\s+)?(?:{prefix_sym_re}\s*)?({number_re})|(?:-\s+)(?:{prefix_sym_re}\s*)?({unsigned_number_re}))(?![\d.,])",
            raw,
        )
        if not m:
            return None
        num_start = m.start(1) if m.group(1) is not None else m.start(2)
        num_end = m.end(1) if m.group(1) is not None else m.end(2)
        tail = raw[num_end:]
        if not tail.strip():
            return None
        # Allow at most one extra number in explanatory text (e.g., "(not divisible by 3)"),
        # but reject if the tail is a continuation expression like "divided by 45 equals 9.09...".
        # Allow a standalone dash used as a text separator (e.g. "50 - скорость...")
        # by only rejecting '-' when it is adjacent to a digit (arithmetic context).
        if re.search(r"[+*/^×÷=]|\d\s*-|-\s*\d", tail):
            return None
        if len(re.findall(number_re, tail)) >= 2:
            return None
        # Require explanatory-like tail: letters or parenthetical tail.
        if re.search(letter_re, tail) or "(" in tail or "\\)" in tail or ")" in tail:
            return (mline.start() + num_start, mline.start() + num_end)
        return None
    return None


def _xslot_leading_number_with_explanation(text: str, number_re: str):
    # Conservative x-slot fallback:
    # accept a short first-line answer like "20 da." but reject reasoning such as
    # "-16 plus 3 would be 20" or equation-like continuations.
    reasoning_cue_re = re.compile(
        r"(?i)\b("
        r"plus|minus|times|multipl(?:y|ied)|divide(?:d)?|"
        r"would|then|because|from|after|before|equals?|"
        r"mas|menos|por|entre|"
        r"més|menys|per|entre|"
        r"plus|moins|fois|divis(?:e|é)|"
        r"pi[uú]s|menos|por|entre|"
        r"gehi|ken|bider|zati|"
        r"جمع|طرح|ضرب|قسمة|"
        r"плюс|минус|умнож|дели|равно"
        r")\b"
    )
    for mline in re.finditer(r"(?m)^.*$", text):
        raw = mline.group(0)
        if not raw.strip():
            continue
        if len(raw.strip()) > 40:
            return None
        nums = list(re.finditer(number_re, raw))
        if len(nums) != 1:
            return None
        m = nums[0]
        if m.start() != re.match(r"\s*", raw).end():
            return None
        tail = raw[m.end():]
        if not tail.strip():
            return None
        if re.search(r"[+\-*/^×÷=]", tail):
            return None
        if reasoning_cue_re.search(tail):
            return None
        if re.search(number_re, tail):
            return None
        if re.search(r"[A-Za-z\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF]", tail) or re.search(
            r"[.،,:;!?。]",
            tail,
        ):
            return (mline.start() + m.start(), mline.start() + m.end())
        return None
    return None


def _extract_answer_label_number(text: str, number_re: str):
    # Language-targeted answer-label pattern (kept strict to avoid prompt echoes).
    label_pat = (
        r"(?:"
        r"final\s*answer|answer|respuesta|resposta|erantzuna|emaitza|ответ|답|答(?:案)?|解|resultado|ergebnis|đáp\s*số|"
        r"الإجابة(?:\s+هي)?|النتيجة|الجواب|"
        r"答案(?:为|是)?|结果(?:为|是)?|運算結果(?:為|是)?|运算结果(?:为|是)?"
        r")"
    )
    pat = re.compile(
        rf"(?is)(?<!\w){label_pat}(?!\w)"
        # Do not let this pre-number chunk consume +/- signs that belong to number.
        rf"[^\n\d+\-−–—]{{0,40}}({number_re})"
    )
    for m in pat.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end(1))
        if line_end < 0:
            line_end = len(text)
        line = text[line_start:line_end]
        line_l = line.lower()
        post_label = text[m.start():line_end]

        if re.search(r"\b(?:operation|equation|following|运算|操作|операц)\b", line_l):
            continue
        if len(line.strip()) > 160:
            continue
        nums_after_label = list(re.finditer(number_re, post_label))
        if len(nums_after_label) == 0:
            continue
        if len(nums_after_label) > 1:
            # Allow extra numeric mention only if it is trailing and parenthetical
            # (e.g., "(or 4.3948 approx)").
            trailing_after_first = post_label[nums_after_label[0].end():]
            if not re.fullmatch(r"\s*\([^()\n]{0,120}\)\s*", trailing_after_first):
                continue

        tail = text[m.end(1):line_end]
        if re.search(r"[+\-*/^×÷]\s*(?:\d|\()", tail):
            continue
        return m.span(1)
    return None


def _single_line_colon_or_equals_number(text: str, number_re: str, generation_ended_with_eos: Optional[bool] = None) -> Optional[Tuple[int, int]]:
    # Non-x fallback: if there is exactly one non-empty line and it has :num or =num,
    # capture that number.
    lines = []
    for m in re.finditer(r"(?m)^.*$", text):
        raw = m.group(0)
        if not raw.strip():
            continue
        # Ignore punctuation-only lines (e.g., "．", "...", "?") so they do not
        # block the single-line fallback.
        if re.fullmatch(r"\s*[.。．，,;；:：!?！？'\"“”‘’`~\-_=+*/×÷|\\()\[\]{}<>《》【】]+\s*", raw):
            continue
        lines.append(m)
    if len(lines) != 1:
        return None
    mline = lines[0]
    raw = mline.group(0)
    ms = list(re.finditer(rf"(?:[:：=≈])\s*({number_re})", raw))
    if not ms:
        return None
    m = ms[-1]
    tail = raw[m.end(1):]
    if re.search(r"[+\-*/^×÷]\s*(?:\d|\()", tail):
        return None
    # If the extracted number is the last number in the output and generation was
    # cut off (not EOS), it is the tail of an unresolved operation chain, not a
    # final answer.
    if generation_ended_with_eos is False and not re.search(r"\d", tail):
        return None
    return (mline.start() + m.start(1), mline.start() + m.end(1))


def _first_line_probable_text_answer(text: str, number_re: str) -> Optional[Tuple[int, int]]:
    # Early answer-like line looks like a textual answer statement and ends in a number.
    cue_re = re.compile(
        r"(?i)\b("
        r"result|answer|antwort|ergebnis|respuesta|resposta|réponse|reponse|résultat|resultado|risultato|erantzuna|emaitza|ответ|"
        r"الإجابة|النتيجة|الجواب|答案|答|結果|结果|"
        r"numerische\s+antwort|numerical\s+answer"
        r")\b"
    )
    seen_nonempty = 0
    max_nonempty = 6
    max_chars = 500
    for mline in re.finditer(r"(?m)^.*$", text):
        raw = mline.group(0)
        if mline.start() > max_chars:
            break
        line = raw.strip()
        if not line:
            continue
        seen_nonempty += 1
        if seen_nonempty > max_nonempty:
            break
        if not cue_re.search(line):
            continue
        # End of line must terminate in a numeric token (with optional punctuation).
        m_end = re.search(rf"(?<![A-Za-z_])({number_re})\s*(?:[%٪]|[.،,:;!?。])?\s*$", raw)
        if not m_end:
            continue
        # Reject if number is preceded or followed by an operator
        num_start_pos = m_end.start(1)
        num_end_pos = m_end.end(1)
        # Check last non-space character before number
        text_before = raw[:num_start_pos].rstrip()
        if text_before and text_before[-1] in '+-×÷x/':
            continue
        # Check character after number
        if num_end_pos < len(raw) and raw[num_end_pos] in '+-×÷x/':
            continue
        return (mline.start() + num_start_pos, mline.start() + num_end_pos)
    return None


def _first_or_last_sentence_promptnums_answer(
    text: str,
    prompt_text: str,
    number_re: str,
    generation_ended_with_eos: Optional[bool] = None,
) -> Optional[Tuple[int, int]]:
    # Fallback: first/last sentence mentions prompt numbers and ends with
    # an extra numeric value interpreted as the answer.
    prompt_nums = set(re.findall(number_re, normalize_unicode_digits(prompt_text or "")))
    if not prompt_nums:
        return None

    # Build candidates from first/last non-empty lines (stable for decimals),
    # plus coarse first/last sentence chunks where sentence punctuation is not
    # part of a decimal number.
    spans = []
    lines = [m for m in re.finditer(r"(?m)^.*$", text) if m.group(0).strip()]
    if lines:
        spans.append((lines[0].start(), lines[0].end()))
        if len(lines) > 1:
            spans.append((lines[-1].start(), lines[-1].end()))

    parts = []
    start = 0
    for m in re.finditer(r"(?<!\d)[.!?。](?!\d)", text):
        end = m.end()
        parts.append((start, end))
        start = end
    if start < len(text):
        parts.append((start, len(text)))
    sent_parts = [(a, b) for a, b in parts if text[a:b].strip()]
    if sent_parts:
        spans.append(sent_parts[0])
        if len(sent_parts) > 1:
            spans.append(sent_parts[-1])

    if not spans:
        return None

    # Deduplicate candidate spans preserving order.
    seen = set()
    candidates = []
    for a, b in spans:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        candidates.append((a, b))

    for a, b in candidates:
        s = text[a:b]
        nums = re.findall(number_re, s)
        if not nums:
            continue
        nums_set = set(nums)
        if not prompt_nums.issubset(nums_set):
            continue
        m_end = re.search(rf"({number_re})\s*(?:[%٪]|[.،,:;!?。])?\s*$", s)
        if not m_end:
            continue
        final_num = m_end.group(1)
        if m_end.start(1) > 0 and s[m_end.start(1) - 1] == "/":
            continue
        # Reject if number is preceded or followed by an operator
        num_start_pos = m_end.start(1)
        num_end_pos = m_end.end(1)
        # Check last non-space character before number
        text_before = s[:num_start_pos].rstrip()
        if text_before and text_before[-1] in '+-x×÷/':
            continue
        # Check character after number
        if num_end_pos < len(s) and s[num_end_pos] in '+-x×÷/':
            continue
        # Require an "extra" final number beyond prompt numbers.
        if final_num in prompt_nums and len(nums_set) <= len(prompt_nums):
            continue
        # If the final number is the last digit in the candidate and generation
        # was cut off (not EOS), it is a truncated repetition, not a final answer.
        if generation_ended_with_eos is False and not re.search(r"\d", s[m_end.end(1):]):
            continue
        return (a + num_start_pos, a + num_end_pos)

    return None


def _extract_cjk_equals_number(text: str, prompt_text: str, number_re: str) -> Optional[Tuple[int, int]]:
    # Conservative Chinese/CJK equality-style answer:
    # "... 等于 33.00", "... 结果是 33.00", "... 答案是 33.00"
    prompt_nums = set(re.findall(number_re, normalize_unicode_digits(prompt_text or "")))
    if not prompt_nums:
        return None

    cue_re = re.compile(r"(?:等于|等於|结果是|結果是|答案是|答案为|答案為)")
    for mline in re.finditer(r"(?m)^.*$", text):
        raw = mline.group(0)
        line = raw.strip()
        if not line:
            continue
        if len(line) > 160:
            continue
        if not cue_re.search(line):
            continue
        nums = re.findall(number_re, line)
        if not nums:
            continue
        nums_set = set(nums)
        if not (prompt_nums & nums_set):
            continue
        m = re.search(rf"(?:等于|等於|结果是|結果是|答案是|答案为|答案為)\s*({number_re})", raw)
        if m is None:
            continue
        tail = raw[m.end(1):]
        if re.search(r"[+\-*/^×÷=]\s*(?:\d|\()", tail):
            continue
        if not re.fullmatch(r"\s*(?:[%٪]|[.،,:;!?。])?\s*(?:这个结果是小数形式的。?)?\s*", tail):
            continue
        return (mline.start() + m.start(1), mline.start() + m.end(1))
    return None


def _build_flexible_expression_pattern(expression: str) -> Optional[str]:
    if not expression:
        return None
    expr = normalize_unicode_digits(expression).strip()
    if not expr:
        return None
    expr_line = expr.splitlines()[0].strip()
    if not expr_line:
        return None

    # Start from escaped literal expression and relax spacing/operator variants.
    pat = re.escape(expr_line)
    pat = pat.replace(r"\ ", r"\s*")
    pat = pat.replace(r"/", r"(?:/|÷)")
    pat = pat.replace(r"÷", r"(?:/|÷)")
    pat = pat.replace(r"\*", r"(?:\*|×)")
    pat = pat.replace(r"×", r"(?:\*|×)")
    return pat


def _extract_expression_equals_number(text: str, expression: str, number_re: str):
    expr_pat = _build_flexible_expression_pattern(expression)
    if not expr_pat:
        return None
    expr_re = re.compile(rf"(?<![\d.,_])({expr_pat})(?![\d.,_])")
    eq_num_re = re.compile(rf"=\s*({number_re})(?![\d.,_A-Za-z])")

    for mexpr in expr_re.finditer(text):
        line_start = text.rfind("\n", 0, mexpr.start()) + 1
        line_end = text.find("\n", mexpr.end())
        if line_end < 0:
            line_end = len(text)

        line = text[line_start:line_end]
        rel_after_expr = mexpr.end() - line_start
        suffix = line[rel_after_expr:]
        eq_matches = list(eq_num_re.finditer(suffix))
        if not eq_matches:
            continue

        last = eq_matches[-1]
        candidate_prefix = suffix[:last.end()]
        # Accept:
        # - direct anchored form: "<expr> = <num>"
        # - chained arithmetic form: "<expr> = ... = <num>"
        # Reject if non-arithmetic text appears between expression and final number.
        if not re.fullmatch(r"[\s=(){}\[\]%٪.,+\-*/^×÷:;!?]+|[\s=(){}\[\]%٪.,+\-*/^×÷:;!?0-9]+", candidate_prefix):
            # Fallback: expression immediately followed by "=<num>" (e.g.
            # "145/5=29(р)-в 1 ящике..."). If the suffix starts directly with
            # "=<number>", trust the first match — the model stated the answer
            # right after the expression before digressing.
            first = eq_matches[0]
            if suffix[:first.start()].strip() in ("", "="):
                # Guard: don't return an intermediate value that is itself
                # followed by an operator (e.g. "= 17 * 5 = 85" → skip 17).
                after_first = suffix[first.end(1):]
                if not re.match(r"\s*[+\-*/^×÷]", after_first):
                    return (
                        line_start + rel_after_expr + first.start(1),
                        line_start + rel_after_expr + first.end(1),
                    )
            continue
        tail = suffix[last.end():]
        if not re.fullmatch(r"\s*(?:[%٪]|[.،,:;!?。])?\s*(?:\([^()\n]{0,60}\))?\s*", tail):
            continue

        return (
            line_start + rel_after_expr + last.start(1),
            line_start + rel_after_expr + last.end(1),
        )
    # High-confidence mode: if we do not find a clean line-ended numeric answer,
    # do not force a fallback match.
    return None


def _first_line_final_equals_number(text: str, number_re: str, prompt_text: str = ""):
    # For mix problems: in first non-empty line, take number after the final "="
    # when there is no operation continuation after that number.
    # Accept trailing punctuation/text after the final number, but avoid
    # degenerate prompt-loop lines with many repeated "=" chunks.
    prompt_nums = set(re.findall(number_re, normalize_unicode_digits(prompt_text or "")))
    for mline in re.finditer(r"(?m)^.*$", text):
        raw = mline.group(0)
        if not raw.strip():
            continue
        if raw.count("=") > 4:
            # Fallback for repeated-computation loops (e.g. mix_0051):
            # find the first "= <num>" where <num> is not a prompt number
            # and is followed by a space + digit (next computation step).
            for mc in re.finditer(rf"=\s*({number_re})", raw):
                if mc.group(1) in prompt_nums:
                    continue
                after = raw[mc.end(1):]
                if re.match(r"\s+\d", after):
                    return (mline.start() + mc.start(1), mline.start() + mc.end(1))
            return None
        ms = list(re.finditer(rf"=\s*({number_re})", raw))
        if not ms:
            return None
        m = ms[-1]
        tail = raw[m.end(1):]
        # Reject mathematical continuation after the captured final number.
        if re.search(r"[+\-*/^×÷]\s*(?:\d|\()", tail):
            return None
        return (mline.start() + m.start(1), mline.start() + m.end(1))
    return None


def _first_line_operation_equals_number(text: str, number_re: str):
    # First non-empty line: if it contains an operation and has a final
    # "= <number>" with no operation continuation after that number, take it.
    for mline in re.finditer(r"(?m)^.*$", text):
        raw = mline.group(0)
        if not raw.strip():
            continue
        if len(raw) > 260:
            return None
        ms = list(re.finditer(rf"=\s*({number_re})", raw))
        if not ms:
            return None
        m = ms[-1]
        left = raw[:m.start()]
        # Require at least one arithmetic operator in the operation text.
        if not re.search(r"[+\-*/×÷]", left):
            return None
        # Reject variable-containing equations (e.g., "x + 20 = 14").
        # Keep optional natural-language prefixes by checking only the segment
        # after the last colon-like separator.
        op_segment = re.split(r"[:：]", left)[-1]
        if re.search(r"[A-Za-z]", op_segment):
            return None
        tail = raw[m.end(1):]
        if re.search(r"[+\-*/^×÷]\s*(?:\d|\()", tail):
            return None
        return (mline.start() + m.start(1), mline.start() + m.end(1))
    return None


def _extract_operation_followed_by_number(text: str, expression: str, number_re: str):
    # Non-x tasks: if original operation appears and is followed by "= number",
    # prefer "=", then fallback to "≈" or ":".
    # allow trailing unit/symbol right after number.
    expr_pat = _build_flexible_expression_pattern(expression)
    if not expr_pat:
        return None

    seps = [r"=", r"≈", r":", r"："]
    for sep in seps:
        pat = re.compile(
            rf"(?<![\d.,_]){expr_pat}(?![\d.,_])\s*{sep}\s*({number_re})"
            rf"(?!\d)"
            rf"(?![.,]\d)"
            rf"(?![A-Za-z_])"
            rf"(?:\s*(?:[%٪]|[^\W\d_]{1,8}|[^\w\s]{1,6}))?"
            rf"(?!\s*[+\-*/^×÷])"
        )
        for m in pat.finditer(text):
            line_end = text.find("\n", m.end(1))
            if line_end < 0:
                line_end = len(text)
            tail = text[m.end(1):line_end]
            # Reject equation continuations like "= 23 * 17 - 1".
            # Strip parenthetical notes (e.g., "(or 48 2/3)") before checking.
            tail_no_parens = re.sub(r'\([^()\n]{0,80}\)', '', tail)
            if re.search(r"[+\-*/^×÷]\s*(?:\d|\()", tail_no_parens):
                continue
            return m.span(1)
    return None


def _extract_mcq(masked: str, number_re: str):
    opt_pat = re.compile(r"(?:^|\\n|\n)\s*([A-Ea-e])\s*[\.\)]\s*(.*?)(?=(?:\\n|\n|$))")
    options = {}
    for mopt in opt_pat.finditer(masked):
        options[mopt.group(1).upper()] = (mopt.start(2), mopt.end(2), mopt.group(2))

    # Inline MCQ options such as "(A) 278 (B) 469 (C) 288".
    if len(options) < 2:
        inline_labels = list(re.finditer(r"\(([A-Ea-e])\)\s*", masked))
        for i, mopt in enumerate(inline_labels):
            seg_start = mopt.end()
            seg_end = inline_labels[i + 1].start() if i + 1 < len(inline_labels) else len(masked)
            segment = masked[seg_start:seg_end]
            nums = list(re.finditer(number_re, segment))
            if not nums:
                continue
            mnum = nums[0]
            label = mopt.group(1).upper()
            options[label] = (
                seg_start + mnum.start(),
                seg_start + mnum.end(),
                mnum.group(0),
            )

    if len(options) < 2:
        return None

    tail = masked[-700:]
    answer_word = (
        r"(?:"
        r"answer|final\s*answer|response|result|"
        r"respuesta|resposta|réponse|reponse|résultat|resultado|risultato|"
        r"antwort|ergebnis|erantzuna|emaitza|"
        r"ответ|результат|"
        r"الإجابة|النتيجة|الجواب|"
        r"答案|结果|結果|答(?:案)?"
        r")"
    )
    choice_patterns = [
        re.compile(r"(?im)[:：=]\s*([A-E])\s*(?:[.。!！?？])?\s*$"),
        re.compile(r"(?im)(?:->|→)\s*([A-E])\s*(?:[.。!！?？])?\s*$"),
        re.compile(r"(?im)\(([A-E])\)\s*(?:[.。!！?？])?\s*$"),
        re.compile(r"(?im)^\s*([A-E])\)\s*(?:[.。!！?？])?\s*$"),
        re.compile(r"(?im)^\s*([A-E])\s*(?:[.。!！?？])?\s*$"),
        re.compile(
            rf"(?im)\b(?:the\s+)?{answer_word}\s+(?:is|=|:)?\s*\(?([A-E])\)?\s*(?:[.。!！?？])?\s*$"
        ),
    ]
    best_choice = None
    for pat in choice_patterns:
        for m in pat.finditer(tail):
            score = m.end()
            if best_choice is None or score > best_choice[0]:
                best_choice = (score, m.group(1).upper())
    chosen = best_choice[1] if best_choice else None
    if not chosen or chosen not in options:
        return None

    opt_start, _, opt_text = options[chosen]
    nums = list(re.finditer(number_re, opt_text))
    if not nums:
        return None
    m = nums[-1]
    return (opt_start + m.start(), opt_start + m.end())


def _extract_variable_assignment(masked: str, prompt_text: str, number_re: str):
    slot_vars = set(v.lower() for v in re.findall(r"(?im)(?:^|\n)\s*([A-Za-z])\s*=\s*$", prompt_text.strip()))
    if not slot_vars:
        slot_vars = {"x"}

    sep = r"(?:=|≈|:|：)"
    for v in sorted(slot_vars):
        pat = re.compile(
            rf"(?i)(?<!\w){re.escape(v)}(?!\w)\s*{sep}\s*({number_re})"
            rf"(?![\d_])"
            rf"(?![.,]\d)"
            rf"(?![A-Za-z])"
            rf"(?![ \t]*[+\-*/^×÷])"
        )
        for m in pat.finditer(masked):
            line_end = masked.find("\n", m.end(1))
            if line_end < 0:
                line_end = len(masked)
            tail = masked[m.end(1) : line_end]
            if re.fullmatch(r"\s*(?:[%٪]|[.،,:;!?。])?\s*", tail):
                return m.span(1)
        # Fallback: chained equalities like "x = 130/5 = 26"
        chain_pat = re.compile(
            rf"(?im)(?<!\w){re.escape(v)}(?!\w)\s*=\s*[^\n]*?=\s*({number_re})\s*(?:[%٪]|[.،,:;!?。])?\s*$"
        )
        m_chain = chain_pat.search(masked)
        if m_chain:
            return m_chain.span(1)
        # Fallback: reversed assignment like "-27 = x"
        rev_pat = re.compile(
            rf"(?im)^\s*({number_re})\s*=\s*{re.escape(v)}(?!\w)\s*(?:[%٪]|[.،,:;!?。])?\s*$"
        )
        m_rev = rev_pat.search(masked)
        if m_rev:
            return m_rev.span(1)
    return None


def _extract_early_variable_assignment(masked: str, prompt_text: str, number_re: str):
    # Conservative fallback for x-slot prompts:
    # look only at the very beginning of output, where the model often emits
    # "<number>\\nx = <number>" before drifting into repeated tasks.
    early_window = masked[:240]
    slot_vars = set(v.lower() for v in re.findall(r"(?im)(?:^|\n)\s*([A-Za-z])\s*=\s*$", prompt_text.strip()))
    if not slot_vars:
        slot_vars = {"x"}

    sep = r"(?:=|≈|:|：)"
    best = None
    for v in sorted(slot_vars):
        pat = re.compile(
            rf"(?i)(?<!\w){re.escape(v)}(?!\w)\s*{sep}\s*({number_re})"
            rf"(?![\d_])"
            rf"(?![.,]\d)"
            rf"(?![A-Za-z])"
            rf"(?![ \t]*[+\-*/^×÷])"
        )
        m = pat.search(early_window)
        if m is None:
            continue
        # Guard against truncating a longer numeric token at the early-window boundary
        # (e.g., window ending inside "-12", which would otherwise match as "-1").
        if m.end(1) == len(early_window) and len(masked) > len(early_window):
            next_ch = masked[len(early_window)]
            if re.match(r"[\dA-Za-z_]", next_ch) or next_ch in {".", ","}:
                continue
        # ensure assignment line ends at the numeric value (optionally punctuation/%)
        abs_num_end = m.end(1)
        line_end = early_window.find("\n", abs_num_end)
        if line_end < 0:
            line_end = len(early_window)
        tail = early_window[abs_num_end:line_end]
        if not re.fullmatch(r"\s*(?:[%٪]|[.،,:;!?。])?\s*", tail):
            continue
        span = m.span(1)
        if best is None or span[0] < best[0]:
            best = span
    return best


def _extract_prompt_echo_variable_assignment(
    masked: str,
    prompt_text: str,
    expression: str,
    number_re: str,
):
    # For x-slot tasks, prefer assignments that follow a prompt-expression echo
    # (e.g., "... 2x + 33 = 91\nx = 29 ...").
    if not expression:
        return None

    expr_pat = _build_flexible_expression_pattern(expression)
    if not expr_pat:
        return None

    slot_vars = set(v.lower() for v in re.findall(r"(?im)(?:^|\n)\s*([A-Za-z])\s*=\s*$", prompt_text.strip()))
    if not slot_vars:
        slot_vars = {"x"}

    expr_re = re.compile(rf"(?<![\d.,_]){expr_pat}(?![\d.,_])")
    sep = r"(?:=|≈|:|：)"

    for em in expr_re.finditer(masked):
        # Only inspect a local window after the echoed expression.
        win_start = em.end()
        win_end = min(len(masked), win_start + 140)
        window = masked[win_start:win_end]

        best = None
        for v in sorted(slot_vars):
            pat = re.compile(
                rf"(?is)(?<!\w){re.escape(v)}(?!\w)\s*{sep}\s*({number_re})"
                rf"(?![\d_])"
                rf"(?![.,]\d)"
                rf"(?![A-Za-z])"
                rf"(?![ \t]*[+\-*/^×÷])"
            )
            m = pat.search(window)
            if m is None:
                continue
            # Ensure line ends at numeric value (optional punctuation/% only).
            abs_num_end = win_start + m.end(1)
            line_end = masked.find("\n", abs_num_end)
            if line_end < 0:
                line_end = len(masked)
            tail = masked[abs_num_end:line_end]
            if not re.fullmatch(r"\s*(?:[%٪]|[.،,:;!?。])?\s*", tail):
                continue
            cand = (win_start + m.start(1), win_start + m.end(1))
            if best is None or cand[0] < best[0]:
                best = cand

        if best is not None:
            return best
    return None


def _extract_boxed_or_bold(masked: str, number_re: str):
    boxed = list(re.finditer(rf"\\boxed\{{\s*({number_re})\s*\}}", masked))
    if boxed:
        return boxed[-1].span(1), "boxed"
    bold = list(re.finditer(rf"\*\*\s*({number_re})\s*\*\*", masked))
    if bold:
        return bold[-1].span(1), "bold"
    return None, None


def _is_prompt_repetition_drift(
    prompt_text: str,
    model_output_raw: str,
    expression: str,
    generation_ended_with_eos: Optional[bool],
) -> bool:
    prompt = normalize_unicode_digits(prompt_text or "")
    output = normalize_unicode_digits(model_output_raw or "")
    if not output.strip():
        return False

    prompt_norm = re.sub(r"\s+", " ", prompt).strip().lower()
    output_norm = re.sub(r"\s+", " ", output).strip().lower()
    has_forum_date_markers = bool(
        re.search(
            r"\bpost\s+by\b|\b(?:mon|tue|wed|thu|fri|sat|sun)\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b",
            output_norm,
            flags=re.IGNORECASE,
        )
    )

    prompt_repeats = 0
    if len(prompt_norm) >= 8:
        prompt_repeats = output_norm.count(prompt_norm)

    expr_repeats = 0
    expr_pat = _build_flexible_expression_pattern(expression or "")
    if expr_pat:
        expr_re = re.compile(rf"(?<![\d.,_]){expr_pat}(?![\d.,_])", re.IGNORECASE)
        expr_repeats = len(list(expr_re.finditer(output)))

    repeated_enough = (prompt_repeats >= 3) or (expr_repeats >= 5)
    if not repeated_enough:
        return False

    # Strictness guard: only classify as repetition drift if, after removing
    # prompt/expression echoes, there is no substantive remaining content.
    residue = output_norm
    if prompt_norm:
        residue = residue.replace(prompt_norm, " ")
        # Remove remaining prompt tokens (e.g., usernames/tags) that can appear
        # between repeated prompt echoes in noisy logs/forums.
        for tok in set(re.findall(r"\w+", prompt_norm, flags=re.UNICODE)):
            if len(tok) >= 2:
                residue = re.sub(rf"\b{re.escape(tok)}\b", " ", residue, flags=re.IGNORECASE)

    if expr_pat:
        expr_re = re.compile(rf"(?<![\d.,_]){expr_pat}(?![\d.,_])", re.IGNORECASE)
        residue = expr_re.sub(" ", residue)
        # Also remove echoed "expression = ?" stubs that are common in prompt loops.
        residue = re.sub(rf"{expr_pat}\s*=\s*\?", " ", residue, flags=re.IGNORECASE)

    # Allow common forum/log timestamp chatter in repetition drift outputs.
    residue = re.sub(r"\bpost\s+by\b", " ", residue, flags=re.IGNORECASE)
    residue = re.sub(
        r"\b(?:mon|tue|wed|thu|fri|sat|sun)\b",
        " ",
        residue,
        flags=re.IGNORECASE,
    )
    residue = re.sub(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b",
        " ",
        residue,
        flags=re.IGNORECASE,
    )
    residue = re.sub(r"\b\d{1,2},", " ", residue)
    residue = re.sub(r"\b\d{1,2}:\d{2}\b", " ", residue)
    residue = re.sub(r"\b(?:am|pm)\b", " ", residue, flags=re.IGNORECASE)
    residue = re.sub(r"\b(?:19|20)\d{2}\b", " ", residue)
    if has_forum_date_markers:
        residue = re.sub(r"\b\d{1,2}\b", " ", residue)

    # Remove punctuation/symbol-only leftovers and whitespace.
    residue = re.sub(r"[\s\?\!\.,;:，。؛:：\(\)\[\]\{\}«»\"'`~\-_=+*/×÷|\\]+", "", residue)
    if residue == "":
        return True

    # Allow tiny truncated tails that are likely cut prompt/expression fragments
    # (e.g., final "1" after many repeats of "150 - 111 = ...").
    expr_flat = "".join(ch for ch in normalize_unicode_digits(expression or "").lower() if ch.isalnum())
    prompt_flat = "".join(ch for ch in normalize_unicode_digits(prompt_text or "").lower() if ch.isalnum())
    if len(residue) <= 2 and (expr_flat.startswith(residue) or prompt_flat.startswith(residue)):
        return True

    # Only for truncated generations (no EOS): allow cut tails that are still
    # prefixes of the prompt/expression echo (can be longer than a few chars).
    if generation_ended_with_eos is False:
        residue_flat = "".join(ch for ch in residue.lower() if ch.isalnum())
        if residue_flat and (expr_flat.startswith(residue_flat) or prompt_flat.startswith(residue_flat)):
            return True

    # If anything substantial remains, this is not pure prompt repetition drift.
    return False


def _is_number_echo_repetition_drift(prompt_text: str, model_output_raw: str) -> bool:
    prompt = normalize_unicode_digits(prompt_text or "")
    output = normalize_unicode_digits(model_output_raw or "")
    if not output.strip():
        return False

    num_pat = re.compile(r"[+-]?\d+(?:[.,]\d+)?")
    prompt_nums = set(num_pat.findall(prompt))
    out_nums = num_pat.findall(output)
    if not out_nums:
        return False
    if not prompt_nums:
        return False
    if not set(out_nums).issubset(prompt_nums):
        return False

    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in re.split(r"\r?\n", output) if ln.strip()]
    if not lines:
        return False
    counts = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    repeated_line = max(counts.values()) >= 5
    x_assign_repeats = len(re.findall(r"(?im)\bx\s*=", output)) >= 5
    eq_repeats = output.count("=") >= 8

    return len(output) >= 120 and (repeated_line or x_assign_repeats or eq_repeats)


def _is_unfinished_operation_repetition(model_output_raw: str) -> bool:
    text = normalize_unicode_digits(model_output_raw or "")
    if not text.strip():
        return False

    unresolved_line = re.compile(
        r"^\s*(?:[+-]?\d+(?:[.,]\d+)?\s*[+\-*/×÷]\s*)+[+-]?\d+(?:[.,]\d+)?\s*=\s*$"
    )
    lines = [ln for ln in re.split(r"\r?\n", text) if ln.strip()]
    if not lines:
        return False

    count = 0
    for ln in lines[:200]:
        if unresolved_line.match(ln):
            count += 1

    return count >= 5


def _is_unfinished_think(model_output_raw: str) -> bool:
    text = normalize_unicode_digits(model_output_raw or "")
    if not text.strip():
        return False
    starts_with_think = bool(re.match(r"(?is)^\s*<think\b[^>]*>", text))
    if not starts_with_think:
        return False
    has_closing = bool(re.search(r"(?is)</think\s*>", text))
    return not has_closing


def _is_text_repetition_drift(
    prompt_text: str,
    expression: str,
    model_output_raw: str,
) -> bool:
    text = normalize_unicode_digits(model_output_raw or "")
    if not text.strip():
        return False

    residue = _remove_prompt_expression_echoes(text, prompt_text or "", expression or "")
    residue = residue.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")

    # Require substantial textual residue.
    tokens = re.findall(r"\w+", residue, flags=re.UNICODE)
    if len(tokens) < 30:
        return False

    # If there is any obvious final-answer signal, do not classify as drift.
    if re.search(r"(?im)\b[a-z]\b\s*(?:=|≈|:|：)\s*[+\-]?\d", residue):
        return False
    if re.search(r"(?im)(?:answer|respuesta|ответ|الإجابة|النتيجة|答案|结果)\s*[:：=]\s*[+\-]?\d", residue):
        return False

    counts = {}
    for t in tokens:
        t = t.lower()
        counts[t] = counts.get(t, 0) + 1
    most_common = max(counts.values()) if counts else 0
    ratio = (most_common / len(tokens)) if tokens else 0.0

    # Also inspect repeated bigrams for looping text fragments.
    bigrams = {}
    for i in range(len(tokens) - 1):
        bg = (tokens[i].lower(), tokens[i + 1].lower())
        bigrams[bg] = bigrams.get(bg, 0) + 1
    max_bigram = max(bigrams.values()) if bigrams else 0

    return (most_common >= 20 and ratio >= 0.18) or (max_bigram >= 10)


def _is_unresolved_operation_loop(model_output_raw: str, generation_ended_with_eos: Optional[bool]) -> bool:
    text = normalize_unicode_digits(model_output_raw or "")
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    if not text.strip():
        return False

    # Arithmetic chain with optional chained equalities:
    # "a op b op c" or "a op b = c op d = e"
    num = r"[+-]?\d+(?:[.,]\d+)?"
    num_atom = rf"(?:{num}|\(\s*{num}\s*\))"
    op = r"[+\-*/×÷]"

    # Guard: repeated solved simple equations (e.g., "24 * 12 = 288") are not
    # unresolved loops by themselves.
    simple_solved_line = re.compile(
        rf"(?im)^\s*{num_atom}\s*{op}\s*{num_atom}\s*=\s*{num_atom}\s*(?:[%٪]|[.،,:;!?。])?\s*$"
    )
    lines = [ln.strip() for ln in re.split(r"\r?\n", text) if ln.strip()]
    solved_simple_count = sum(1 for ln in lines if simple_solved_line.match(ln))
    if solved_simple_count >= 3:
        unresolved_signature = re.compile(
            rf"(?:{num_atom}\s*{op}\s*){{2,}}{num_atom}|"
            rf"{num_atom}\s*{op}\s*{num_atom}\s*=\s*(?:{num_atom}\s*{op}\s*)+{num_atom}"
        )
        if not unresolved_signature.search(text):
            return False

    chain = (
        rf"(?:{num_atom}\s*{op}\s*)+{num_atom}"
        rf"(?:\s*=\s*(?:{num_atom}\s*{op}\s*)*{num_atom})*"
    )
    chain_re = re.compile(chain)

    matches = list(chain_re.finditer(text))
    if not matches:
        return False

    # Count operators in largest chain.
    longest = max(matches, key=lambda m: (m.end() - m.start()))
    frag = longest.group(0)
    op_count = len(re.findall(op, frag))

    # Repetition signal: repeated arithmetic fragments.
    normalized_frags = [
        re.sub(r"\s+", " ", m.group(0).strip())
        for m in matches
        if len(m.group(0).strip()) >= 20
    ]
    rep_count = 0
    if normalized_frags:
        counts = {}
        for f in normalized_frags:
            counts[f] = counts.get(f, 0) + 1
        rep_count = max(counts.values())

    # Strong loop signatures.
    if rep_count >= 3 and op_count >= 3:
        return True
    if frag.count("=") >= 6 and op_count >= 10:
        return True
    if op_count >= 30 and "=" not in frag:
        return True
    if op_count >= 20 and generation_ended_with_eos is False:
        return True

    # Repeated prompt-like operation lines, e.g.:
    # "Compute: 1000 - 999 =", "احسب: 1000 - 998 =", ...
    line_pat = re.compile(
        rf"(?im)^\s*(?:[^\n\d]{{1,120}}[:：]\s*)?{num_atom}\s*{op}\s*{num_atom}\s*=\s*$"
    )
    op_prefix_pat = re.compile(
        rf"(?im)^\s*(?:[^\n\d]{{1,120}}[:：]\s*)?{num_atom}\s*{op}\s*{num_atom}\s*="
    )
    lines = [ln.strip() for ln in re.split(r"\r?\n", text) if ln.strip()]
    op_lines = [ln for ln in lines if line_pat.match(ln)]
    if len(op_lines) >= 6:
        return True
    # More tolerant fallback for prompt+operation loops with long prefixes:
    # requires ":" + operator + trailing "=" and repeated lines.
    loose_op_line = re.compile(r"(?i)^[^=\n]{0,260}[:：][^=\n]*[+\-*/×÷][^=\n]*=\s*$")
    loose_lines = [re.sub(r"\s+", " ", ln) for ln in lines if loose_op_line.match(ln)]
    if len(loose_lines) >= 6:
        counts = {}
        for ln in loose_lines:
            counts[ln] = counts.get(ln, 0) + 1
        if max(counts.values()) >= 2:
            return True
        # Also accept monotonic prompt+operation streams with mostly unique lines,
        # especially when generation appears truncated.
        if len(loose_lines) >= 10 and generation_ended_with_eos is False:
            return True
    # Repeated operation templates with non-numeric placeholder RHS after "="
    # e.g., "计算： 120 - 110 = <u>...</u>．"
    placeholder_rhs = 0
    for ln in lines:
        m = op_prefix_pat.match(ln)
        if not m:
            continue
        rhs = ln[m.end():]
        if not re.search(r"\d", rhs):
            placeholder_rhs += 1
    if placeholder_rhs >= 6:
        return True

    # Short prompt-variation loop: lines with digits ending with "= ?" placeholder
    # (e.g. "Kalkulatu: 332 + 191 = ?", "Kalkulatu: 332 x 191 = ?").
    question_rhs = sum(
        1 for ln in lines
        if re.search(r"=\s*[?？]\s*$", ln) and re.search(r"\d", ln)
    )
    if question_rhs >= 2:
        return True

    return False


def _is_xslot_unresolved_operation_loop(
    prompt_text: str,
    model_output_raw: str,
    generation_ended_with_eos: Optional[bool],
) -> bool:
    text = normalize_unicode_digits(model_output_raw or "")
    if not text.strip():
        return False

    slot_vars = set(v.lower() for v in re.findall(r"(?im)(?:^|\n)\s*([A-Za-z])\s*=\s*$", (prompt_text or "").strip()))
    if not slot_vars:
        slot_vars = {"x"}

    num = r"[+-]?\d+(?:[.,]\d+)?"
    frac = r"[+-]?\d+\s*/\s*\d+"
    clean_rhs = rf"(?:{num}|{frac})"

    def _contains_slot_var(fragment: str, var_name: str) -> bool:
        return re.search(rf"(?i)(?<![A-Za-z]){re.escape(var_name)}(?![A-Za-z])", fragment) is not None

    unresolved_lines = []
    var_op_lines = []
    clean_count = 0

    for raw in re.split(r"\r?\n", text):
        line = raw.strip()
        if not line:
            continue
        matched_var = None
        for v in sorted(slot_vars):
            if re.match(rf"(?i)^\s*{re.escape(v)}\s*=", line):
                matched_var = v
                break
        if matched_var is None:
            # Also track non-assignment variable-operation lines like "2x + 34 - 30"
            # for x-slot unresolved loops.
            for v in sorted(slot_vars):
                if _contains_slot_var(line, v) and re.search(r"[+\-*/×÷=]", line):
                    var_op_lines.append(re.sub(r"\s+", " ", line))
                    break
            continue

        if re.match(rf"(?i)^\s*{re.escape(matched_var)}\s*=\s*{clean_rhs}\s*(?:[%٪]|[.،,:;!?。])?\s*$", line):
            clean_count += 1
            continue

        rhs = re.sub(rf"(?i)^\s*{re.escape(matched_var)}\s*=\s*", "", line).strip()
        if re.search(r"[+\-*/×÷=]", rhs) or re.search(r"(?i)\b[a-z]\b", rhs):
            unresolved_lines.append(re.sub(r"\s+", " ", line))
        # Assignment with variable-containing RHS (e.g., "x = 2x + 4").
        if _contains_slot_var(rhs, matched_var):
            var_op_lines.append(re.sub(r"\s+", " ", line))
            if rhs.count("=") >= 3:
                return True

    if not unresolved_lines and not var_op_lines:
        return False

    counts = {}
    for ln in unresolved_lines:
        counts[ln] = counts.get(ln, 0) + 1
    repeated_same = max(counts.values()) if counts else 0
    unresolved_count = len(unresolved_lines)
    var_op_count = len(var_op_lines)
    var_counts = {}
    for ln in var_op_lines:
        var_counts[ln] = var_counts.get(ln, 0) + 1
    repeated_var_same = max(var_counts.values()) if var_counts else 0

    if repeated_same >= 3 and unresolved_count >= max(4, clean_count * 2):
        return True
    if unresolved_count >= 8 and unresolved_count >= clean_count * 2:
        return True
    # Variable-operation loops, including changing lines such as
    # "x = 2x + 4", "x = 3x + 1", or repeated "x = 2x + 4".
    if repeated_var_same >= 3 and var_op_count >= max(4, clean_count * 2):
        return True
    if var_op_count >= 8 and var_op_count >= clean_count * 2:
        return True
    if generation_ended_with_eos is False and unresolved_count >= 4 and unresolved_count >= clean_count * 2:
        return True
    if generation_ended_with_eos is False and var_op_count >= 4 and var_op_count >= clean_count * 2:
        return True

    return False


def _last_line_eq_number_if_completed(
    text: str,
    number_re: str,
    generation_ended_with_eos: Optional[bool],
    slot_vars=None,
):
    # Conservative fallback: use only when generation is known to have ended on EOS.
    if generation_ended_with_eos is not True:
        return None
    lines = [ln for ln in re.split(r"\r?\n", text) if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    if slot_vars:
        assigned_vars = set(
            v.lower()
            for v in re.findall(r"(?i)(?<!\w)([A-Za-z])\s*(?:=|≈|:|：)\s*", last)
        )
        if assigned_vars and not assigned_vars.issubset({v.lower() for v in slot_vars}):
            return None
    ms = list(re.finditer(rf"=\s*({number_re})", last))
    if not ms:
        return None
    m = ms[-1]
    tail = last[m.end(1):]
    if re.search(r"[+\-*/^×÷]\s*(?:\d|\()", tail):
        return None
    if not re.fullmatch(r"\s*(?:[%٪]|[.،,:;!?。])?\s*", tail):
        return None

    # Map back to absolute span.
    off = text.rfind(last)
    if off < 0:
        return None
    return (off + m.start(1), off + m.end(1))


def _is_tail_number_candidate(text: str, span: Tuple[int, int]) -> bool:
    _, end = span
    tail = text[end:]
    return re.fullmatch(r"\s*(?:[%٪]|[.،,:;!?。])?\s*", tail or "") is not None


def _last_standalone_number_line(
    text: str,
    number_re: str,
    generation_ended_with_eos: Optional[bool],
) -> Optional[Tuple[int, int]]:
    # Conservative fallback for non-x prompts:
    # accept a trailing standalone number line only if generation ended cleanly
    # OR there is an explicit completion marker (e.g., </think>) before it.
    lines = list(re.finditer(r"(?m)^.*$", text))
    if not lines:
        return None

    for i in range(len(lines) - 1, -1, -1):
        mline = lines[i]
        raw = mline.group(0)
        if not raw.strip():
            continue
        m = re.fullmatch(rf"\s*({number_re})\s*(?:[%٪]|[.،,:;!?。])?\s*", raw)
        if not m:
            return None
        prefix = text[: mline.start()]
        has_completion_marker = bool(re.search(r"(?i)</think>|final\s*answer|answer\s*:", prefix[-400:]))
        if generation_ended_with_eos is True or has_completion_marker:
            return (mline.start() + m.start(1), mline.start() + m.end(1))
        return None
    return None


def _remove_prompt_expression_echoes(text: str, prompt_text: str, expression: str) -> str:
    residue = text
    p = normalize_unicode_digits(prompt_text or "")
    if p.strip():
        # Remove full prompt and its non-empty lines (common prompt echoes/drift).
        residue = residue.replace(p.strip(), " ")
        for ln in re.split(r"\r?\n", p):
            ln = ln.strip()
            if len(ln) >= 4:
                residue = residue.replace(ln, " ")

    expr_pat = _build_flexible_expression_pattern(expression or "")
    if expr_pat:
        residue = re.sub(expr_pat, " ", residue)

    return residue


def _is_truncated_unanswered_prompt_loop(
    model_output_raw: str, generation_ended_with_eos: Optional[bool]
) -> bool:
    """Return True when generation was cut off (no EOS) and virtually all
    non-empty output lines are unanswered operation-prompt lines: they contain
    an arithmetic operator and end with '=' but have no numeric result after it.
    This pattern means the model looped over prompt-template variations and was
    truncated before ever producing an answer; any numbers in those lines belong
    to the echoed expression, not to a real answer."""
    if generation_ended_with_eos is not False:
        return False

    text = normalize_unicode_digits(model_output_raw or "")
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    if not text.strip():
        return False

    lines = [ln.strip() for ln in re.split(r"\r?\n", text) if ln.strip()]
    if len(lines) < 5:
        return False

    # Completed: '=' is followed by a number somewhere on the line.
    completed_re = re.compile(r"=\s*[+-]?\d")
    # Unanswered: has an arithmetic operator and ends with '=' (no result).
    # Also treat ':' and 'x' as operators when they appear between two digits
    # (e.g. "120 : 10 =" or "120 x 10 =") since some locales/models write
    # division and multiplication that way.
    unanswered_op_re = re.compile(
        r"^[^\n]*(?:[+\-*/×÷]|\d\s*[x:]\s*\d)[^\n]*=\s*$"
    )

    unanswered = 0
    other = 0
    for ln in lines:
        if completed_re.search(ln):
            other += 1
        elif unanswered_op_re.match(ln):
            unanswered += 1
        else:
            other += 1

    if unanswered < 5:
        return False

    # Require that the vast majority of lines are unanswered op lines.
    return other <= max(2, int(unanswered * 0.15))


def extract_high_confidence_only(
    prompt_text: str,
    model_output_raw: str,
    expression: str = "",
    prompt_variant: str = "",
    problem_type: str = "",
    generation_ended_with_eos: Optional[bool] = None,
):
    prompt_text = normalize_unicode_digits(prompt_text or "")
    model_output_raw = normalize_unicode_digits(model_output_raw or "")
    masked = model_output_raw.replace("−", "-").replace("–", "-").replace("—", "-")
    norm_prompt = prompt_text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    norm = masked.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    # When the output is a single line of CJK text with "。" as sentence
    # separators (no newlines), split at "。 " so line-based extractors work.
    if "\n" not in norm and "\u3002" in norm:
        norm = norm.replace("\u3002 ", "\u3002\n")
        masked = masked.replace("\u3002 ", "\u3002\n")
    # Similarly, treat ". " as a sentence boundary when no newlines are present,
    # but only when the period is NOT preceded by a digit (to avoid splitting
    # decimal numbers like "3.14 (m)").
    if "\n" not in norm and re.search(r"\D\. \S", norm):
        norm = re.sub(r"(?<=\D)\. (?=\S)", ".\n", norm)
        masked = re.sub(r"(?<=\D)\. (?=\S)", ".\n", masked)
    number_re = r"[+-]?(?:\d+(?:[.,]\d+)?|\d{1,3}(?:[.,_ ]\d{3})+(?:[.,]\d+)?|[.,]\d+)"
    number_re_fraction = r"[+-]?\d+/\d+"
    slot_vars = set(v.lower() for v in re.findall(r"(?im)(?:^|\n)\s*([A-Za-z])\s*=\s*$", norm_prompt.strip()))
    asks_var_value = bool(slot_vars)

    # 0) Literal empty output special case.
    if model_output_raw == "":
        return [], "empty", False

    # 0.25) Unfinished CoT block.
    if _is_unfinished_think(model_output_raw):
        return [], "unfinished_think", False

    # 0.5) Pre-compute drift flags (used as fallback after all extraction steps).
    is_prompt_repetition_drift = _is_prompt_repetition_drift(
        norm_prompt,
        model_output_raw,
        expression or "",
        generation_ended_with_eos,
    )
    is_number_echo_repetition_drift = _is_number_echo_repetition_drift(norm_prompt, model_output_raw)

    # 1) No digits gate
    if not re.search(r"\d", masked):
        return [], "no_digit_gate", False
    # 1.1) Also no_digit_gate if digits only come from prompt/expression echoes.
    residue_wo_echo = _remove_prompt_expression_echoes(masked, norm_prompt, expression or "")
    if not re.search(r"\d", residue_wo_echo):
        if not asks_var_value and (is_prompt_repetition_drift or is_number_echo_repetition_drift):
            return [], "prompt_repetition_drift", False
        return [], "no_digit_gate", False

    # 1.2) Mix problems: first-line final '= number' is highest priority.
    if (problem_type or "").strip().lower() == "mix":
        mix_span = _first_line_final_equals_number(norm, number_re, norm_prompt)
        if mix_span is not None:
            raw_span = _map_norm_span_to_raw(model_output_raw, norm, mix_span)
            if raw_span is None:
                a, b = mix_span
                return [norm[a:b]], "mix_first_line_final_eq", False
            a, b = raw_span
            return [model_output_raw[a:b]], "mix_first_line_final_eq", False

    # 1.3) Non-x: exact prompt expression followed by '= number' has priority
    # over leading standalone numbers.
    if not asks_var_value:
        expr_eq_span = _extract_expression_equals_number(masked, expression or "", number_re)
        if expr_eq_span is not None:
            if _is_tail_number_candidate(masked, expr_eq_span) and generation_ended_with_eos is False:
                expr_eq_span = None
        if expr_eq_span is not None:
            a, b = expr_eq_span
            return [model_output_raw[a:b]], "prompt_expression_eq_number", False

    # 1.5) Optional preference for "answer" prompts:
    # prefer explicit "<expression> = <number>" before leading-number grabs.
    if (prompt_variant or "").strip().lower() == "answer":
        expr_span = _extract_expression_equals_number(masked, expression or "", number_re)
        if expr_span is not None:
            if _is_tail_number_candidate(masked, expr_span) and generation_ended_with_eos is False:
                expr_span = None
        if expr_span is not None:
            a, b = expr_span
            return [model_output_raw[a:b]], "expression_eq_number_answer_variant", False

    # 1.7) Non-x: original operation followed by number (allow unit/symbol).
    if not asks_var_value:
        op_span = _extract_operation_followed_by_number(masked, expression or "", number_re)
        if op_span is not None:
            if _is_tail_number_candidate(masked, op_span) and generation_ended_with_eos is False:
                op_span = None
        if op_span is not None:
            a, b = op_span
            return [model_output_raw[a:b]], "operation_followed_by_number", False

    # 1.8) Non-x: if the entire output contains exactly one number and it is not
    # a prompt number, use it (e.g. "約 25.88" with prompt "1118 / 43 =").
    if not asks_var_value:
        all_num_matches = list(re.finditer(number_re, masked))
        if len(all_num_matches) == 1:
            prompt_nums = set(re.findall(number_re, norm_prompt))
            if all_num_matches[0].group(0) not in prompt_nums:
                a, b = all_num_matches[0].span()
                return [model_output_raw[a:b]], "single_number_output", False

    # 2) Early non-variable first-line standalone number (wins before MCQ)
    if not asks_var_value:
        lead_span = _leading_standalone_number_line(norm, number_re, norm_prompt)
        if lead_span is not None:
            raw_span = _map_norm_span_to_raw(model_output_raw, norm, lead_span)
            if raw_span is None:
                a, b = lead_span
                return [norm[a:b]], "leading_standalone_number", False
            a, b = raw_span
            return [model_output_raw[a:b]], "leading_standalone_number", False

    # 3) Prefix-leading number on first line (non-variable)
    if not asks_var_value:
        pref_span = _leading_prefix_number(norm)
        if pref_span is not None:
            raw_span = _map_norm_span_to_raw(model_output_raw, norm, pref_span)
            if raw_span is None:
                a, b = pref_span
                return [norm[a:b]], "leading_prefix_number", False
            a, b = raw_span
            return [model_output_raw[a:b]], "leading_prefix_number", False

    # 4) Leading number with unit/explanation on first line (non-variable)
    if not asks_var_value:
        unit_span = _leading_number_with_unit_explanation(norm, number_re, norm_prompt)
        if unit_span is not None:
            raw_span = _map_norm_span_to_raw(model_output_raw, norm, unit_span)
            if raw_span is None:
                a, b = unit_span
                return [norm[a:b]], "leading_number_unit_expl", False
            a, b = raw_span
            return [model_output_raw[a:b]], "leading_number_unit_expl", False

    # 4.5) CJK equality-style answer sentence (non-variable)
    if not asks_var_value:
        cjk_eq_span = _extract_cjk_equals_number(masked, norm_prompt, number_re)
        if cjk_eq_span is not None:
            a, b = cjk_eq_span
            return [model_output_raw[a:b]], "probable_text_answer", False

    # 4.75) Leading number with non-math explanation text (non-variable)
    if not asks_var_value:
        expl_span = _leading_number_with_explanation(norm, number_re)
        if expl_span is not None:
            raw_span = _map_norm_span_to_raw(model_output_raw, norm, expl_span)
            if raw_span is None:
                a, b = expl_span
                return [norm[a:b]], "leading_number_expl", False
            a, b = raw_span
            return [model_output_raw[a:b]], "leading_number_expl", False

    # 5) MCQ
    mcq_span = _extract_mcq(masked, number_re)
    if mcq_span is not None:
        a, b = mcq_span
        return [model_output_raw[a:b]], "mcq", False

    # 6) Strict x-slot handling
    if asks_var_value:
        lead_span = _leading_standalone_number_line(norm, number_re, norm_prompt)
        if lead_span is None:
            lead_span = _leading_prefix_number(norm)
        lead_value = None
        if lead_span is not None:
            raw_span = _map_norm_span_to_raw(model_output_raw, norm, lead_span)
            if raw_span is None:
                a, b = lead_span
                lead_value = norm[a:b]
            else:
                a, b = raw_span
                lead_value = model_output_raw[a:b]

        echo_var_span = _extract_prompt_echo_variable_assignment(
            masked=masked,
            prompt_text=norm_prompt,
            expression=expression or "",
            number_re=number_re,
        )
        var_value = None
        if echo_var_span is not None:
            a, b = echo_var_span
            var_value = model_output_raw[a:b]

        if var_value is None:
            early_var_span = _extract_early_variable_assignment(
                masked=masked,
                prompt_text=norm_prompt,
                number_re=number_re,
            )
            if early_var_span is not None:
                a, b = early_var_span
                var_value = model_output_raw[a:b]

        if var_value is None:
            any_var_span = _extract_variable_assignment(masked, norm_prompt, number_re)
            if any_var_span is not None:
                a, b = any_var_span
                var_value = model_output_raw[a:b]

        # Fraction fallback for x-slot: only if no integer/decimal assignment was found.
        if var_value is None:
            echo_var_span = _extract_prompt_echo_variable_assignment(
                masked=masked,
                prompt_text=norm_prompt,
                expression=expression or "",
                number_re=number_re_fraction,
            )
            if echo_var_span is not None:
                a, b = echo_var_span
                var_value = model_output_raw[a:b]
        if var_value is None:
            early_var_span = _extract_early_variable_assignment(
                masked=masked,
                prompt_text=norm_prompt,
                number_re=number_re_fraction,
            )
            if early_var_span is not None:
                a, b = early_var_span
                var_value = model_output_raw[a:b]
        if var_value is None:
            any_var_span = _extract_variable_assignment(masked, norm_prompt, number_re_fraction)
            if any_var_span is not None:
                a, b = any_var_span
                var_value = model_output_raw[a:b]

        # If leading standalone exists and we also have explicit assignment,
        # include both options as requested.
        if lead_value is not None and var_value is not None:
            if lead_value == var_value:
                return [var_value], "xslot_leading_and_assignment", False
            return [lead_value, var_value], "xslot_leading_and_assignment", False

        # If no leading standalone but direct x assignment exists, take it.
        if var_value is not None:
            return [var_value], "xslot_assignment_only", False

        # Else fallback to leading standalone only.
        if lead_value is not None:
            return [lead_value], "leading_standalone_number_xslot", False

        # Conservative x-slot fallback for short answer-like first lines such as "20 da."
        xslot_expl_span = _xslot_leading_number_with_explanation(norm, number_re)
        if xslot_expl_span is not None:
            raw_span = _map_norm_span_to_raw(model_output_raw, norm, xslot_expl_span)
            if raw_span is None:
                a, b = xslot_expl_span
                return [norm[a:b]], "leading_number_expl", False
            a, b = raw_span
            return [model_output_raw[a:b]], "leading_number_expl", False

        txt_ans = _first_line_probable_text_answer(masked, number_re)
        if txt_ans is not None:
            a, b = txt_ans
            return [model_output_raw[a:b]], "probable_text_answer", False

        if is_prompt_repetition_drift or is_number_echo_repetition_drift:
            return [], "prompt_repetition_drift", False

        # X-slot unresolved loop fallback (only after all x-slot extraction attempts fail).
        if _is_xslot_unresolved_operation_loop(norm_prompt, model_output_raw, generation_ended_with_eos):
            return [], "unresolved_operation_loop", False

    # 6.5) Truncated output whose lines are all unanswered prompt-echo operations.
    # When generation was cut off (no EOS) and virtually every non-empty line is
    # an unanswered operation template (has an operator, ends with "=", no numeric
    # result), the model looped over prompt variations and never produced an answer.
    # Numbers present in those lines come from the echoed expression, not from a
    # real answer.  This guard runs before answer-label extraction (step 8) and
    # applies to all problem types, including mix/eq.
    if _is_truncated_unanswered_prompt_loop(model_output_raw, generation_ended_with_eos):
        return [], "unresolved_operation_loop", False

    # 7) Boxed / bold
    marker_span, marker_name = _extract_boxed_or_bold(masked, number_re)
    if marker_span is not None:
        a, b = marker_span
        return [model_output_raw[a:b]], marker_name, False

    # 8) Answer label (e.g., "Đáp số: 1728")
    ans_lbl = _extract_answer_label_number(masked, number_re)
    if ans_lbl is not None:
        a, b = ans_lbl
        return [model_output_raw[a:b]], "answer_label", False

    if not asks_var_value:
        one_line = _single_line_colon_or_equals_number(masked, number_re, generation_ended_with_eos)
        if one_line is not None:
            a, b = one_line
            return [model_output_raw[a:b]], "answer_label", False

    # 8.5) First-line textual answer ending in a number (non-x fallback).
    if not asks_var_value:
        txt_ans = _first_line_probable_text_answer(masked, number_re)
        if txt_ans is not None:
            a, b = txt_ans
            return [model_output_raw[a:b]], "probable_text_answer", False
        sent_ans = _first_or_last_sentence_promptnums_answer(masked, norm_prompt, number_re, generation_ended_with_eos)
        if sent_ans is not None:
            a, b = sent_ans
            return [model_output_raw[a:b]], "sentence_prompt_numbers_answer", False

    # 9) Unfinished operation repetition (infinite unresolved "a op b =" pattern).
    if _is_unfinished_operation_repetition(model_output_raw):
        return [], "unfinished_operation_repetition", False

    # 10) Conservative last-line equation-end fallback, only if generation ended on EOS.
    eos_tail_span = _last_line_eq_number_if_completed(
        masked,
        number_re,
        generation_ended_with_eos,
        slot_vars=slot_vars if asks_var_value else None,
    )
    if eos_tail_span is not None:
        a, b = eos_tail_span
        return [model_output_raw[a:b]], "last_line_eq_number_eos_guard", False

    # 11) Non-x fallback: last standalone number line (e.g., final answer after </think>).
    if not asks_var_value:
        tail_num_span = _last_standalone_number_line(masked, number_re, generation_ended_with_eos)
        if tail_num_span is not None:
            a, b = tail_num_span
            return [model_output_raw[a:b]], "last_standalone_number_line", False

    # 11.5) Low-priority fallback: first-line operation ending with '= number'
    # for numeric-only operation segment (no variables).
    first_eq_span = _first_line_operation_equals_number(norm, number_re)
    if first_eq_span is not None:
        raw_span = _map_norm_span_to_raw(model_output_raw, norm, first_eq_span)
        if raw_span is None:
            a, b = first_eq_span
            return [norm[a:b]], "first_line_equal", False
        a, b = raw_span
        return [model_output_raw[a:b]], "first_line_equal", False

    # 12) Drift / loop classifiers — only after all concrete extractors have failed.
    if not asks_var_value and (is_prompt_repetition_drift or is_number_echo_repetition_drift):
        return [], "prompt_repetition_drift", False
    # For non-mix/eq types, apply stricter loop and drift checks.
    # Mix/eq types allow intermediate calculations, so skip these.
    if not asks_var_value and (problem_type or "").strip().lower() not in ("mix", "eq"):
        if _is_unresolved_operation_loop(model_output_raw, generation_ended_with_eos):
            return [], "unresolved_operation_loop", False
        if _is_text_repetition_drift(norm_prompt, expression or "", model_output_raw):
            return [], "other_repetition_drift", False

    return [], "", True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        default="results",
        help="Input results root (existing JSONLs, as written by evaluation/run_eval.py).",
    )
    ap.add_argument(
        "--out-dir",
        default="results/raw/base_and_instruct",
        help="Output root (new JSONLs, does NOT overwrite input).",
    )
    ap.add_argument("--models", default="", help="Optional comma-separated model folder name filters.")
    ap.add_argument(
        "--sample-like-annotator",
        action="store_true",
        help="Only apply regex re-extraction to the same subset selected by sample_examples().",
    )
    ap.add_argument(
        "--sample-jsonl",
        default="",
        help="Exact JSONL sample file to rerun on (e.g. regex_analysis/manual_regex_reextraction_sample_550.jsonl).",
    )
    ap.add_argument(
        "--out-jsonl",
        default="",
        help="Optional single JSONL output path. Intended for sample reruns.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pt-c", type=int, default=3)
    ap.add_argument("--pt-w", type=int, default=3)
    ap.add_argument("--pt-n", type=int, default=4)
    ap.add_argument("--it-c", type=int, default=1)
    ap.add_argument("--it-w", type=int, default=1)
    ap.add_argument("--it-n", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes. Parallelism is by input JSONL file.",
    )
    ap.add_argument(
        "--only-sampled-rows",
        action="store_true",
        help="When --sample-like-annotator is enabled, write only sampled rows (drop all others).",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_filters = {m.strip() for m in args.models.split(",") if m.strip()}
    sampled_keys = None
    num_workers = max(1, args.num_workers)

    if args.sample_like_annotator and args.sample_jsonl:
        raise ValueError("Use either --sample-like-annotator or --sample-jsonl, not both.")
    if args.out_jsonl and not sampled_keys and not args.sample_like_annotator and not args.sample_jsonl:
        raise ValueError("--out-jsonl is intended for sample reruns; provide --sample-like-annotator or --sample-jsonl.")

    if args.sample_like_annotator:
        selected, _shortages = sample_examples(args)
        sampled_keys = {_record_key_from_sampled_row(row) for row in selected}
        print(f"[INFO] sample_like_annotator enabled: selected={len(sampled_keys)}")
    elif args.sample_jsonl:
        sample_jsonl = Path(args.sample_jsonl)
        if not sample_jsonl.exists():
            raise FileNotFoundError(f"Sample JSONL not found: {sample_jsonl}")
        sampled_keys = _load_sample_keys_from_jsonl(sample_jsonl)
        print(f"[INFO] sample_jsonl enabled: selected={len(sampled_keys)} from {sample_jsonl}")

    single_jsonl_mode = bool(args.out_jsonl)
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else None
    if single_jsonl_mode:
        if not args.only_sampled_rows:
            raise ValueError("--out-jsonl requires --only-sampled-rows.")
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed_files = 0
    processed_rows = 0

    input_files = list(iter_jsonl_files(in_dir))
    tasks = [
        (
            str(in_path),
            str(in_dir),
            str(out_dir),
            model_filters,
            sampled_keys,
            args.only_sampled_rows,
            single_jsonl_mode,
        )
        for in_path in input_files
    ]

    print(f"[INFO] files={len(input_files)} workers={num_workers} host_cpus={os.cpu_count()}")

    single_jsonl_lines = []
    if num_workers == 1:
        for task in tasks:
            wrote_any, rows, lines = _process_file(task)
            if wrote_any:
                processed_files += 1
            processed_rows += rows
            if lines:
                single_jsonl_lines.extend(lines)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for wrote_any, rows, lines in ex.map(_process_file, tasks):
                if wrote_any:
                    processed_files += 1
                processed_rows += rows
                if lines:
                    single_jsonl_lines.extend(lines)

    if out_jsonl is not None:
        with out_jsonl.open("w", encoding="utf-8") as f:
            for line in single_jsonl_lines:
                f.write(line)

    print(f"Done. Processed files={processed_files}, rows={processed_rows}")
    print(f"Output written to: {out_jsonl if out_jsonl is not None else out_dir}")
    if model_filters and processed_files == 0:
        print(
            "No files matched --models filter. "
            "Filters are matched against model folder names using case-insensitive normalized substring matching."
        )


if __name__ == "__main__":
    main()
