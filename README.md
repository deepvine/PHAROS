# PHAROS — Goal-Conditioned First-Action Prior for LLM Agents

> Mitigating direction bias in KnowAgent-style ICL agents through a minimal, anchoring-free strategic prior.

PHAROS extends [KnowAgent (Zhu et al., NAACL 2025 Findings)](https://arxiv.org/abs/2403.03101) by adding a single LLM call that produces a **3-line strategic prior** before the forward agent executes. The prior contains:

```
Question type: <one short label>
Expected answer form: <specific form of the final answer>
First action: <Retrieve | Search | Either>
Reason: <one-sentence justification>
```

This prior is injected as `[STRATEGIC PRIOR]` into the existing KnowAgent forward prompt. The forward agent remains autoregressive — the prior acts only as a soft prior on the first action while preserving step-level autonomy on subsequent steps (avoiding *anchoring*).

## What's the bias?

KnowAgent's in-context examples (one starting with `Retrieve`, one with `Search`) suggest a 50/50 first-action distribution, but in practice we measure **Search-First Rate (SFR) of 0.58–0.84** across five benchmarks — biased toward `Search` even on questions where targeted Wikipedia `Retrieve` is more efficient. PHAROS exposes this with the SFR metric and proposes a prompt-level mitigation.

## Highlights

- **No fine-tuning** — single extra LLM call per task
- **Drop-in to KnowAgent prompting track** — does not modify the forward agent or action graph
- **Direction-bias diagnostic (SFR)** — quantifies first-action distortion
- **5 benchmarks** — HotpotQA (easy/medium/hard), 2WikiMultihopQA, StrategyQA
- **Bing v7 deprecation handled** — Tavily Search integrated as raw-snippet replacement

## Repo Structure

```
.
├── experiments/                # PHAROS extensions (new code)
│   ├── backward_prompts.py     # prompt templates + file loader
│   ├── prompts/                # textual prompt definitions
│   │   └── janus_v3_minimal.txt
│   ├── backward_agent.py       # strategic-prior generator
│   ├── bidirectional_agent.py  # PHAROS / Baseline / BackwardOnly agents
│   ├── run_experiment.py       # main CLI runner
│   ├── bias_metrics.py         # SFR, FAE, etc.
│   ├── benchmark_loaders.py    # 2Wiki, StrategyQA loaders
│   └── results/real/           # experimental result jsonls
│
├── Path_Generation/            # KnowAgent base (modified files only annotated)
│   └── hotpotqa_run/
│       ├── agent_arch.py       # [MODIFIED] Tavily, single-call forward, oracle
│       ├── config.py           # [MODIFIED] dotenv, TAVILY_API_KEY
│       └── llms.py             # [MODIFIED] gpt-4.1-mini support
│
├── paper_acl_ko/               # ACL-style LaTeX paper (Korean)
└── REPORT.md                   # detailed analysis (Korean)
```

## Setup

```bash
git clone https://github.com/deepvine/PHAROS.git
cd PHAROS
pip install -r requirements.txt
```

Create `.env`:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

(Tavily provides a free tier of 1k search queries/month — sufficient for the experiments below.)

## Run

```bash
# Baseline (KnowAgent*) + PHAROS, n=50, HotpotQA easy
python -m experiments.run_experiment \
  --benchmark hotpotqa --split easy --n 50 \
  --bp-mode minimal --skip-backonly

# Other splits
python -m experiments.run_experiment --benchmark hotpotqa --split medium --n 50 --bp-mode minimal --skip-backonly
python -m experiments.run_experiment --benchmark hotpotqa --split hard   --n 50 --bp-mode minimal --skip-backonly

# 2WikiMultihopQA
python -m experiments.run_experiment --benchmark 2wikimultihopqa --split validation --n 50 --bp-mode minimal --skip-backonly

# StrategyQA
python -m experiments.run_experiment --benchmark strategyqa --split train --n 50 --bp-mode minimal --skip-backonly

# Retrieve-oracle ablation (force first action = Retrieve)
python -m experiments.run_experiment --benchmark hotpotqa --split easy --n 50 \
  --force-retrieve-first --skip-backonly --skip-bidir
```

## Editing the prompt

PHAROS reads its strategic-prior prompt from a plain text file:

```
experiments/prompts/janus_v3_minimal.txt
```

Edit this file directly — no code change needed. The prompt is loaded at runtime via `experiments/backward_prompts.py::_load_prompt`.

## Results (n=50 per split, gpt-4.1-mini, Tavily Search active)

| Benchmark | KA* EM | PHAROS EM | ΔEM strict |
|-----------|--------|-----------|-----------|
| HotpotQA easy | 0.380 | 0.280 | −0.10 |
| HotpotQA medium | 0.340 | **0.440** | **+0.10** |
| HotpotQA hard | 0.280 | 0.160 | −0.12 |
| 2WikiMultihopQA | 0.260 | 0.140 | −0.12 |
| StrategyQA | 0.120 | **0.420** | **+0.30** |

PHAROS shows the strongest gains in commonsense yes/no reasoning (StrategyQA) where the *Expected answer form* hint normalises verbose answers into the strict-EM target. On Wikipedia-grounded multi-hop tasks the gain is mixed; see [REPORT.md](REPORT.md) for analysis of context-budget trade-offs and failure modes.

## Citation

If you use PHAROS, please cite both KnowAgent and this work:

```bibtex
@article{zhu2024knowagent,
  title={KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents},
  author={Zhu, Yuqi and Qiao, Shuofei and others},
  journal={arXiv preprint arXiv:2403.03101},
  year={2024}
}

@misc{pharos2026,
  title={{PHAROS}: Goal-Conditioned First-Action Prior for LLM Agents},
  author={(your name)},
  year={2026},
  howpublished={\url{https://github.com/deepvine/PHAROS}}
}
```

## License

This project builds on KnowAgent (Apache 2.0). PHAROS extensions in `experiments/` and modifications to `Path_Generation/` are released under the same license. See [LICENSE.txt](LICENSE.txt).

## Acknowledgements

- KnowAgent authors (Zhu et al.) for the action-graph framework and ICL prompting baseline.
- Tavily for the search API used as a Bing-v7 replacement.
