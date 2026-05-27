# format_gate_multilingual

This repository contains the code, input data and results for the paper "Format Before Math: A Behavioral and Mechanistic Study of LLMs’ Language-Induced Failures in Arithmetic".

## Requirements
Run commands from the repository root unless noted otherwise.

`requirements.txt` contains the lean runtime dependency list. `requirements-full.txt` preserves the original full environment freeze.

```bash
pip install -r requirements.txt
pip install -e .
```

To recreate the original full environment instead:

```bash
pip install -r requirements-full.txt
pip install -e .
```

## Repository Structure
- `data/` — benchmark problems and few-shot examples
- `evaluation/` — model evaluation scripts
- `analysis/` — behavioral and mechanistic analysis
- `results/` — raw and aggregated outputs
- `paper_figures/` — figures in the paper

## Reproducing Results

### Data
Includes the arithmetic problems (600 instances, without prompts) and the few-shot data, as well as the scripts to create them.

### Evaluation
Includes:
- The scripts to run the zero- and few-shot evaluation on base or instructed models and the constrained evaluation (only digits allowed, minus sign, space, and EOS).
- The `evaluation/extractor.py` script to score and categorize outputs.
- The launch scripts that exemplify how to run the evaluation and extraction scripts.


### Behavioral Analysis
- First, to get the summarized results (the CSV files in results/aggregated), run:
```bash
python analysis/behavioral/summarize_results.py --results-dir results/raw/base_and_instruct --out-dir results/aggregated/summarized_results --skip-consistency
```

- To print the accuracy per problem type for the selected models:
```bash
python analysis/behavioral/summarize_problem_type.py "results/aggregated/base_and_instruct/accuracy_by_type.csv" gemma-3-4b-pt Meta-Llama-3.2-3B Qwen3-1.7B-Base
```

- To get a CSV with model-level metrics for tables 12 and 13 (Appendix) in the paper:
```bash
python analysis/behavioral/summarize_model_table.py --summarized-dir results/aggregated/base_and_instruct --out table.csv
```

- To plot the heatmap where y axis are languages and x axis are prompts / problem types (Figure 5 in paper, "gemma-3-4b-pt_vs_Meta-Llama-3.2-3B_vs_Qwen3-1.7B-Base_by-prompt_no_answer_rate.png" in paper_figures):
```bash
python analysis/behavioral/plot_no_answer_heatmap.py \
    --csv results/aggregated/base_and_instruct/answer_types_by_type.csv \
    --model gemma-3-4b-pt Meta-Llama-3.2-3B Qwen3-1.7B-Base \
    --output paper_figures \
    --by-prompt
```
Use `--by-prompt` for prompts on the x axis, or `--all-prompts` for problem types.

- To create the scatter plot (Figure 1 in the paper, "accuracy_scatter_by_family.png" in paper_figures) of the accuracy distribution per (lang, prompt), do:
```bash
python analysis/behavioral/plot_accuracy_scatter.py \
  --zs-results results/aggregated/base_and_instruct/accuracy_by_prompt.csv \
  --fs-results results/aggregated/fewshot/accuracy_by_prompt.csv \
  --out-dir paper_figures
```
  
- To create the bar plot with the answer type distribution (Figure 2 in the paper, "plot_answer_types.png" in paper_figures)
```bash
python analysis/behavioral/plot_answer_types.py \
  --zs-summarized-dir results/aggregated/base_and_instruct \
  --fs-summarized-dir results/aggregated/fewshot \
  --output paper_figures/plot_answer_types.png
```



### Mechanistic Analysis

#### First token
- To build a table summarizing the first token for each record, saving its language, prompt, and answer type:
```bash
python analysis/first_token/first_token_table.py --model model_name --results-dir results/raw/base_and_instruct --output first_token_analysis/first_token_table_model_name.csv
```

- To print (in terminal) a summary of the previous table:
```bash
python analysis/first_token/analyze_first_token.py first_token_analysis/first_token_table_model_name.csv
```

#### Cosine similarities
- To compute cosine similarities between hidden states at the "=" token (last token) across all (lang, prompt pairs) and save the results in parquet:
```bash
python analysis/cosine/compute_cosine_similarities.py \
    --model-path /path/to/model \
    --results-dir results/raw/base_and_instruct \
    --output cosine_sims_model_name.parquet
```
    
- Aggregate info in parquet to create the cosine curves plot (Figure 6 in paper, figures curves_* in paper_figures). This script reads a parquet produced by compute_cosine_similarities.py.
```bash
python analysis/cosine/aggregate_cosine_similarities.py \
    --input cosine_sims/model_name.parquet \
    --output cosine_sims_agg/model_name/ \
    --curves
```

#### Logit lens
- For each transformer layer, project the hidden state at the last "=" token through the final LayerNorm + LM head, and record
the top-K most probable tokens. To create the top 1 prediction at each layer for the selected (lang, prompt):

```bash
python analysis/logit_lens/logit_lens.py \
  --model-path /path/to/model \
  --results-dir  results/raw/model_name \
  --lang ar \
  --prompt answer \
  --top-k 1 \
  --output logit_lens_results/to_analyse/model_name/ar_answer_top1.csv
```

- Once the previous script has been run for the selected (lang, prompt) setups, create a summarized/pivot CSV split by category: ANSWER vs. NO-ANSWER. The fields in the pivot CSV are: token, total_count, then for each layer L{layer}_count and L{layer}_pct.
```bash
python analysis/logit_lens/layer_token_frequencies.py logit_lens_results/to_analyse/model_name \
    --pivot logit_lens_results/to_analyse/split_by_categories/model_name/pivot.csv \
    --split-answer-category
```

- To plot the top 1 logit lens results separated by ANSWER vs NO-ANSWER cases (Figures 3 and 7 in the paper and in logit_lens folder inside paper_figures). The argument min-pct specifies the minimum...

```bash
python analysis/logit_lens/plot_layer_token_frequencies.py \
  --split-folder logit_lens/to_analyse/split_by_categories/model_name \
  --split-out logit_lens_result/to_analyse/model_name_answer.png \
             logit_lens_result/to_analyse/model_name_no_answer.png \
             logit_lens_result/to_analyse/model_name_legend.png \
  --min-pct 20
```


#### Steering

- The following is the script to create a steering vector based on the given positive (POS) and negative (NEG) features of results and evaluate on negative features instances. The steering vector is computed on 100 instances and the evaluation run on other 100 instances. This is an example on how to run the script for applying the steering vector at all generation positions. Results are saved by default on "steering_outputs" folder.
```bash
python analysis/steering/steering_vectors.py \
  --model_base_dir /path/to/hf-models \
  --results_dir results/raw/base_and_instruct \
  --fs_results_dir results/raw/fewshot \
  --output_dir steering_outputs \
  --vec_dir steering_outputs \
  --models "gemma-3-4b-pt" \
  --common_features "language=es,prompt=answer" \
  --positive_features "answer_type=ANSWER-C" \
  --negative_features "answer_type=ANSWER-I"
```
- To do the same but only at the "=" position; results are saved by default on "steering_outputs_eq_pos" folder.
```bash
python analysis/steering/steering_vectors_eq_pos.py \
  --model_base_dir /path/to/hf-models \
  --results_dir results/raw/base_and_instruct \
  --fs_results_dir results/raw/fewshot \
  --output_dir steering_outputs_eq_pos \
  --vec_dir steering_outputs \
  --models "gemma-3-4b-pt" \
  --common_features "language=es,prompt=answer" \
  --positive_features "answer_type=ANSWER-C" \
  --negative_features "answer_type=ANSWER-I"
```
- The bash script `run_steering_vectors.sh` can be used to run several jobs of steering vectors.
- The following script summarizes the results of several steering vector jobs.
```bash
python analysis/steering/summarize_steering_outputs.py --steering-dir steering_outputs --out-dir steering_summaries
```
- To plot the results as in Figure 4 in the paper (folder steering within paper_figures)
```bash
python analysis/steering/plot_steering_results.py neg.csv pos.csv --extra extra.csv -o out.png
```
Here `neg.csv` and `pos.csv` point to the CSV output of executing a steering vector in the positive direction (coeff 1) and in the negative direction (coeff -1 and evaluation set is the correct-feature set). `--extra` can point to an extra CSV, for example the all-position negative steering vector.

### Results
Divided into raw results and aggregated summaries, each split into base_and_instruct, constrained_decoding, and fewshot. The raw files are organized by model, then language, then a JSONL file per prompt type. Due to their size (~4.5 GB), **raw results are hosted on OSF: https://osf.io/ymuvw**. The aggregated results (CSV files, created with the analysis/behavioral scripts) are included directly in `results/aggregated/`.

## Data Description

### File Formats
| Format | Extension | Used for |
|--------|-----------|----------|
| JSONL | `.jsonl` | Benchmark dataset, few-shot examples, raw evaluation results (one JSON object per line) |
| CSV | `.csv` | Aggregated metrics, behavioral summaries, first-token and logit-lens analysis outputs |
| Parquet | `.parquet` | Cosine similarity analysis outputs (generated by analysis scripts, not stored in the repository) |
| PNG | `.png` | Paper figures (`paper_figures/`) |

### Benchmark Dataset (`data/`)
- `simple_math_dataset.jsonl` — 600 arithmetic problem instances.
- `simple_math_fewshot.jsonl` — 300 few-shot examples (same schema).

Both files share the following schema:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique problem identifier (e.g. `sum_0091`) |
| `type` | str | Problem type: `sum`, `sub`, `mul`, `div`, or `mix` |
| `subtype` | str | Problem subtype (e.g. `sum_ab`, `mix_p0`) |
| `expression` | str | Arithmetic expression (e.g. `"216 + 305"`) |
| `answer` | str | Correct numeric answer |
| `meta` | object | Generation metadata: operand values and flags |

### Raw Evaluation Results
Due to their size (~4.5 GB), raw evaluation results are hosted on OSF: **https://osf.io/ymuvw**

They are organized as `{split}/{model}/{language}/{prompt}.jsonl`, where:
- **split**: `base_and_instruct`, `fewshot`, or `constrained_decoding`
- **model**: HuggingFace model name
- **language**: language code (see table below)
- **prompt**: prompt variant (`answer`, `compute`, `determine`, `long_numeric`, `solve`)

Each record extends the benchmark schema with evaluation fields:

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | Model identifier |
| `language` | str | Language code of the prompt |
| `prompt_variant` | str | Prompt template name |
| `prompt_text` | str | Full prompt fed to the model |
| `generation_kwargs` | object | Decoding parameters (temperature, max tokens, etc.) |
| `input_ids` | list[int] | Tokenized prompt |
| `output_ids` | list[int] | Tokenized model output |
| `model_output_raw` | str | Raw model generation |
| `extracted_answer` | list | Answer candidates found by the extractor |
| `contains_gold_label` | bool | Whether the correct answer was extracted |
| `extractor` | str | Extraction rule that matched |

The aggregated results (CSV files) produced by the behavioral analysis scripts are included directly in the repository under `results/aggregated/`.

### Languages
Nine natural languages plus two control conditions:

| Code | Language |
|------|----------|
| `ar` | Arabic |
| `ca` | Catalan |
| `de` | German |
| `en` | English |
| `es` | Spanish |
| `eu` | Basque |
| `fr` | French |
| `ru` | Russian |
| `zh` | Chinese |
| `empty` | No language instruction (control) |
| `fake` | Invented/fake language script (control) |

### Models Evaluated
30 publicly available models from HuggingFace Hub, grouped by family:

| Family | Models |
|--------|--------|
| Qwen3 | `Qwen3-0.6B-Base`, `Qwen3-0.6B`, `Qwen3-1.7B-Base`, `Qwen3-1.7B`, `Qwen3-4B-Base`, `Qwen3-4B`, `Qwen3-8B-Base`, `Qwen3-8B`, `Qwen3-14B-Base`, `Qwen3-14B` |
| Gemma-3 | `gemma-3-4b-pt`, `gemma-3-4b-it`, `gemma-3-12b-pt`, `gemma-3-12b-it` |
| Llama-3 | `Llama-3.2-1B`, `Llama-3.2-1B-Instruct`, `Meta-Llama-3.2-3B`, `Llama-3.2-3B-Instruct`, `Meta-Llama-3.1-8B`, `Meta-Llama-3.1-8B-Instruct` |
| EuroLLM | `EuroLLM-1.7B`, `EuroLLM-9B`, `EuroLLM-1.7B-Instruct`, `EuroLLM-9B-Instruct` |
| Salamandra | `salamandra-2b`, `salamandra-7b`, `salamandra-2b-instruct`, `salamandra-7b-instruct` |
| Teuken | `Teuken-7B-base-v0.6`, `Teuken-7B-instruct-v0.6` |

### Access and Reuse
All data and code are released under the MIT License (see `LICENSE`). The benchmark dataset was procedurally generated — no third-party data or personal information is included. The models used for evaluation are publicly available on HuggingFace Hub under their respective licenses.

## Citation
Citation pending.
