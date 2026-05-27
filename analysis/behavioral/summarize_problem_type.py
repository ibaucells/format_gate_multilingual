#!/usr/bin/env python3
"""
Computes mean accuracy per problem type for specified models,
averaged across all (language, prompt) combinations.

Usage:
    python summarize_accuracy.py input.csv output.csv model1 model2 ...

If no models are specified, all models in the file are used.

python summarize_problem_type.py "../../results/aggregated/base_and_instruct/accuracy_by_type.csv" gemma-3-4b-pt Meta-Llama-3.2-3B Qwen3-1.7B-Base
"""

import sys
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("Usage: python summarize_accuracy.py input.csv output.csv [model1 model2 ...]")
        sys.exit(1)

    input_path = sys.argv[1]
    selected_models = sys.argv[2:] if len(sys.argv) > 2 else None

    df = pd.read_csv(input_path)

    if selected_models:
        missing = [m for m in selected_models if m not in df["model"].unique()]
        if missing:
            print(f"Warning: these models were not found in the data: {missing}")
        df = df[df["model"].isin(selected_models)]

    # Average accuracy across all (language, prompt) combinations for each (model, type)
    summary = (
        df.groupby(["model", "type"])["accuracy"]
        .mean()
        .reset_index()
    )

    # Pivot so rows = type, columns = model
    table = summary.pivot(index="type", columns="model", values="accuracy")
    table.columns.name = None
    table.index.name = "type"
    table = table.round(4)

    type_order = ["sum", "sub", "mul", "div", "mix", "eq"]
    table = table.reindex([t for t in type_order if t in table.index])

    # Print table to stdout instead of creating a CSV file.
    print(table.to_string())


if __name__ == "__main__":
    main()
