import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class Example:
    id: str
    type: str              # sum, sub, mul, div, mix, eq
    subtype: str           # e.g., mix_p0, eq_axpb_eq_c
    expression: str        # the math expression shown to the model
    answer: str            # canonical numeric answer (string)
    meta: Dict             # extra metadata (operands, has_parentheses, etc.)

def _make_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:04d}"
    
_INT_RE = re.compile(r"-?\d+")

def _answer_appears_in_expression(expr: str, ans: str) -> bool:
    # True if the answer appears as a number token anywhere in the expression
    return ans in _INT_RE.findall(expr)

def load_existing_expressions(jsonl_path: str) -> set:
    exprs = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            exprs.add(rec["expression"])
    return exprs


def write_jsonl(path: str, examples: List[Example]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "id": ex.id,
                "type": ex.type,
                "subtype": ex.subtype,
                "expression": ex.expression,
                "answer": ex.answer,
                "meta": ex.meta
            }, ensure_ascii=False) + "\n")

def generate_sum(
    n: int, rng: random.Random, start_idx: int = 0,
    lo: int = 0, hi: int = 500, canonicalize: bool = True,
    max_tries: int = 100000
) -> List[Example]:
    exs: List[Example] = []
    seen = set()
    tries = 0

    while len(exs) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("generate_sum: max_tries exceeded. Increase range or max_tries.")

        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if canonicalize and a > b:
            a, b = b, a

        key = (a, b)
        if key in seen:
            continue

        expr = f"{a} + {b}"
        ans = str(a + b)
        
        if _answer_appears_in_expression(expr, ans):
            continue
        
        seen.add(key)
        
        exs.append(Example(
            id=_make_id("sum", start_idx + len(exs)),
            type="sum",
            subtype="sum_ab",
            expression=expr,
            answer=ans,
            meta={"a": a, "b": b, "canonicalized": canonicalize}
        ))
    return exs

def generate_sub(
    n: int, rng: random.Random, start_idx: int = 0,
    lo: int = 0, hi: int = 500, max_tries: int = 100000
) -> List[Example]:
    exs: List[Example] = []
    seen = set()
    tries = 0

    while len(exs) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("generate_sub: max_tries exceeded. Increase range or max_tries.")

        a = rng.randint(lo, hi)
        b = rng.randint(0, a)  # ensures non-negative result
        
        if b == a:
            continue

        key = (a, b)
        if key in seen:
            continue

        expr = f"{a} - {b}"
        ans = str(a - b)
        
        if _answer_appears_in_expression(expr, ans):
            continue
            
        seen.add(key)
            
        exs.append(Example(
            id=_make_id("sub", start_idx + len(exs)),
            type="sub",
            subtype="sub_ab",
            expression=expr,
            answer=ans,
            meta={"a": a, "b": b}
        ))
    return exs

def generate_mul(
    n: int, rng: random.Random, start_idx: int = 0,
    lo: int = -30, hi: int = 30, canonicalize: bool = True,
    max_tries: int = 100000
) -> List[Example]:
    exs: List[Example] = []
    seen = set()
    tries = 0

    while len(exs) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("generate_mul: max_tries exceeded. Increase range or max_tries.")

        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if canonicalize and a > b:
            a, b = b, a

        key = (a, b)
        if key in seen:
            continue

        expr = f"{a} * {b}"
        ans = str(a * b)
        
        if _answer_appears_in_expression(expr, ans):
            continue
        
        seen.add(key)
            
        exs.append(Example(
            id=_make_id("mul", start_idx + len(exs)),
            type="mul",
            subtype="mul_ab",
            expression=expr,
            answer=ans,
            meta={"a": a, "b": b, "canonicalized": canonicalize}
        ))
    return exs

def generate_div(
    n: int, rng: random.Random, start_idx: int = 0,
    divisor_lo: int = 1, divisor_hi: int = 50,
    quot_lo: int = 0, quot_hi: int = 50,
    max_tries: int = 100000
) -> List[Example]:
    """Integer division only: a / b where a = b*q."""
    exs: List[Example] = []
    seen = set()
    tries = 0

    while len(exs) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("generate_div: max_tries exceeded. Increase range or max_tries.")

        b = rng.randint(divisor_lo, divisor_hi)
        q = rng.randint(quot_lo, quot_hi)
        a = b * q

        key = (a, b)
        if key in seen:
            continue

        expr = f"{a} / {b}"
        ans = str(q)
        
        if _answer_appears_in_expression(expr, ans):
            continue
            
        seen.add(key)
            
        exs.append(Example(
            id=_make_id("div", start_idx + len(exs)),
            type="div",
            subtype="div_int",
            expression=expr,
            answer=ans,
            meta={"a": a, "b": b, "q": q}
        ))
    return exs

MIX_PATTERNS = [
    ("mix_p0", "a + b * c", False, lambda a, b, c: a + b * c),
    ("mix_p1", "(a + b) * c", True,  lambda a, b, c: (a + b) * c),
    ("mix_p2", "a * b - c", False, lambda a, b, c: a * b - c),
    ("mix_p3", "a * (b - c)", True,  lambda a, b, c: a * (b - c)),
]

def generate_mix(
    n: int, rng: random.Random, start_idx: int = 0,
    a_lo: int = 0, a_hi: int = 30,
    b_lo: int = 0, b_hi: int = 30,
    c_lo: int = 0, c_hi: int = 20,
    allow_parentheses: bool = True,
    max_tries: int = 200000
) -> List[Example]:
    exs: List[Example] = []
    seen = set()
    tries = 0

    patterns = MIX_PATTERNS if allow_parentheses else [p for p in MIX_PATTERNS if not p[2]]

    while len(exs) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("generate_mix: max_tries exceeded. Increase ranges or max_tries.")

        a = rng.randint(a_lo, a_hi)
        b = rng.randint(b_lo, b_hi)
        c = rng.randint(c_lo, c_hi)

        pid, template, has_paren, fn = rng.choice(patterns)
        expr = template.replace("a", str(a)).replace("b", str(b)).replace("c", str(c))

        key = (pid, expr)
        if key in seen:
            continue

        ans = str(fn(a, b, c))
        
        if _answer_appears_in_expression(expr, ans):
            continue
            
        seen.add(key)
            
        exs.append(Example(
            id=_make_id("mix", start_idx + len(exs)),
            type="mix",
            subtype=pid,
            expression=expr,
            answer=ans,
            meta={"a": a, "b": b, "c": c, "has_parentheses": has_paren}
        ))
    return exs

def generate_eq(
    n: int, rng: random.Random, start_idx: int = 0,
    x_lo: int = -50, x_hi: int = 50,
    a_choices: Tuple[int, ...] = (-5,-4,-3,-2,-1,1,2,3,4,5),
    b_lo: int = -50, b_hi: int = 50,
    max_tries: int = 200000,
    append_x_line: bool = True,
) -> List[Example]:
    """
    Linear equation: a*x + b = c, generated from x.

    If append_x_line=True, expression becomes:
        "<equation>\nx"
    which creates a language-agnostic answer slot.
    """
    exs: List[Example] = []
    seen = set()
    tries = 0

    while len(exs) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("generate_eq: max_tries exceeded. Increase ranges or max_tries.")

        x = rng.randint(x_lo, x_hi)
        a = rng.choice(a_choices)
        b = rng.randint(b_lo, b_hi)
        c = a * x + b
        if b == 0:
            continue

        # Convert 1x -> x and -1x -> -x
        if a == 1:
            left = "x"
        elif a == -1:
            left = "-x"
        else:
            left = f"{a}x"

        left += f" {'+' if b > 0 else '-'} {abs(b)}"

        eq_line = f"{left} = {c}"
        expr = eq_line + ("\nx" if append_x_line else "")

        key = expr  # uniqueness on displayed expression
        if key in seen:
            continue

        ans = str(x)
        
        if ans == str(b) or ans == str(c):
            continue
        
        if _answer_appears_in_expression(expr, ans):
            continue
            
        seen.add(key)
            
        exs.append(Example(
            id=_make_id("eq", start_idx + len(exs)),
            type="eq",
            subtype="eq_axpb_eq_c" + ("__xslot" if append_x_line else ""),
            expression=expr,
            answer=ans,
            meta={"a": a, "b": b, "c": c, "x": x, "append_x_line": append_x_line}
        ))
    return exs

def main(
    allow_parentheses_in_mix: bool = True,
    append_x_line_in_eq: bool = True,
    eval_jsonl_path: str = "simple_math_dataset.jsonl",
    fewshot_out_path: str = "simple_math_fewshot.jsonl",
    fewshot_seed: int = 123,
    fewshot_k: int = 5,
):
    existing_exprs = load_existing_expressions(eval_jsonl_path)

    rng_fs = random.Random(fewshot_seed)

    fewshot: List[Example] = []

    def keep_new(exs: List[Example]) -> List[Example]:
        return [ex for ex in exs if ex.expression not in existing_exprs]

    oversample = 10

    cand = keep_new(generate_sum(fewshot_k * oversample, rng_fs, start_idx=0, canonicalize=True))
    if len(cand) < fewshot_k:
        raise RuntimeError("Not enough non-overlapping SUM examples. Increase oversample/ranges.")
    fewshot += cand[:fewshot_k]

    cand = keep_new(generate_sub(fewshot_k * oversample, rng_fs, start_idx=0))
    if len(cand) < fewshot_k:
        raise RuntimeError("Not enough non-overlapping SUB examples. Increase oversample/ranges.")
    fewshot += cand[:fewshot_k]

    cand = keep_new(generate_mul(fewshot_k * oversample, rng_fs, start_idx=0, canonicalize=True))
    if len(cand) < fewshot_k:
        raise RuntimeError("Not enough non-overlapping MUL examples. Increase oversample/ranges.")
    fewshot += cand[:fewshot_k]

    cand = keep_new(generate_div(fewshot_k * oversample, rng_fs, start_idx=0))
    if len(cand) < fewshot_k:
        raise RuntimeError("Not enough non-overlapping DIV examples. Increase oversample/ranges.")
    fewshot += cand[:fewshot_k]

    cand = keep_new(generate_mix(fewshot_k * oversample, rng_fs, start_idx=0, allow_parentheses=allow_parentheses_in_mix))
    if len(cand) < fewshot_k:
        raise RuntimeError("Not enough non-overlapping MIX examples. Increase oversample/ranges.")
    fewshot += cand[:fewshot_k]

    cand = keep_new(generate_eq(fewshot_k * oversample, rng_fs, start_idx=0, append_x_line=append_x_line_in_eq))
    if len(cand) < fewshot_k:
        raise RuntimeError("Not enough non-overlapping EQ examples. Increase oversample/ranges.")
    fewshot += cand[:fewshot_k]

    fs_exprs = [ex.expression for ex in fewshot]
    if len(set(fs_exprs)) != len(fs_exprs):
        raise RuntimeError("Few-shot set contains duplicate expressions; increase oversample/ranges/seed.")

    write_jsonl(fewshot_out_path, fewshot)
    print(f"Wrote {len(fewshot)} few-shot examples to {fewshot_out_path}")

if __name__ == "__main__":
    main(
        eval_jsonl_path="simple_math_dataset.jsonl",
        fewshot_out_path="simple_math_fewshot.jsonl",
        fewshot_seed=123,
        fewshot_k=50
    )

