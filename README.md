# Same System, Different Verdict

Code, materials and raw results for a pre-registered conjoint audit of how large language models classify AI systems under the EU AI Act (submitted to ACIIDS 2027).

## Design

- **960 fixed AI-system descriptions** (`descriptions_960.csv`). Each one varies risk profile, company name style, country and company size, and has a reference class.
- **2 prompting arms**: `direct` (classify immediately) and `checker` (answer questions Q1–Q5 on Art. 5, Annex I, Annex III, Art. 6(3) and Art. 50 in order, then give the final class). The prompts are in `prompts.json`.
- **EU AI Act excerpts** are given to every model as context (`ai_act_excerpts.txt`).
- **10% repeats** to measure test–retest consistency, chosen with a fixed seed (2026).
- That gives 960 × 2 + 192 = **2,112 requests per model**.
- Output classes: `prohibited`, `high_risk`, `transparency_only`, `minimal`, plus the five intermediate answers `q1`–`q5`.

## Models

| Key | Model ID | Developer | Route |
|---|---|---|---|
| `qwen3` | `qwen/qwen3-32b` | Alibaba Qwen (China) | OpenRouter |
| `nemotron` | `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA (US) | NVIDIA API (449 requests via OpenRouter `:free` endpoint) |
| `xing` | `xingchen-agi/xing4.0-29b-a4b:free` | China Telecom TeleAI (China) | OpenRouter |
| `gemini_lite` | `google/gemini-2.5-flash-lite` | Google (US) | OpenRouter |

The `model_requested` and `model_returned` columns record the model ID used for every request.

## Results summary

| Model | Requests | Valid output | Accuracy vs. reference | Accuracy: direct | Accuracy: checker |
|---|---|---|---|---|---|
| Qwen3 32B | 2,112 | 100.0% | 80.3% | 79.7% | 80.8% |
| Nemotron 3 Ultra | 2,112 | 100.0% | 77.3% | 78.3% | 76.4% |
| Gemini 2.5 Flash-Lite | 2,112 | 99.9% | 61.1% | 64.6% | 57.6% |
| Xing 4.0 | 2,112 | 100.0% | 56.9% | 68.7% | 44.9% |

## Files

```
run_experiment.py        experiment runner (job list, seed, API calls, parsing, summary)
config.json              models and run settings
descriptions_960.csv     the 960 descriptions with reference classes
prompts.json             system prompt and the two prompting arms
ai_act_excerpts.txt      EU AI Act excerpts given to the models
requirements.txt         Python packages
.env.example             template for API keys
results/
  <model>.jsonl          raw record per request (job ID = description|arm|repeat, raw output, parsed answer, tokens, timestamp)
  results_all.csv        all models merged, one row per request (8,448 rows)
```

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env        # then add your OpenRouter / NVIDIA keys
python run_experiment.py --pilot     # 20 requests per model
python run_experiment.py --full      # 2,112 requests per model (resumable)
python run_experiment.py --summary   # rebuilds results/results_all.csv
```

Finished requests are skipped on re-runs, and requests that failed with an error are retried. Running `--summary` on the included `.jsonl` files rebuilds `results_all.csv` exactly.
