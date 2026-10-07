# RAG Pipeline for Malaysia AI Policy Q&A

A Retrieval-Augmented Generation (RAG) assistant that answers questions about Malaysia's AI governance policies. Answers are grounded only in the retrieved document chunks and include numbered citations. When retrieval confidence is low, the question is escalated to a human reviewer.

BAA5123 AI and Decision Making, Topic 18: Retrieval-Augmented Assistants for Internal Knowledge.

## How it works

```
PDFs (data/) → page text (pypdf) → 1,200-char chunks, 200 overlap
  → BAAI/bge-small-en-v1.5 embeddings → FAISS cosine index
  → top-5 retrieval → Claude answers from those chunks only, with citations [1], [2], ...
```

- **Generator:** Claude Haiku 4.5 (default) or Claude Opus 5.5, set with `ANTHROPIC_MODEL`
- **Evaluation:** RAGAS (faithfulness, answer relevancy, context precision), with Claude Sonnet 5.5 as the judge
- **Confidence signal:** the cosine similarity of the top-ranked chunk (`top1_score`). If it falls below the threshold, the question goes to a human.

## Corpus

| Document | Pages | Chunks |
|---|---:|---:|
| AI Boardroom (Malaysia) | 38 | 73 |
| AI Code of Ethics (Malaysia) | 23 | 31 |
| Stanford AI Index Report 2026 | 424 | 1,005 |
| National AI Action Plan | 93 | 162 |
| **Total** | **578** | **1,271** |

Total size is 1,095,072 characters. The Stanford report accounts for 79% of the chunks, which matters for cross-document questions (see [Known issues](#known-issues)).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
```

## Usage

```bash
python rag_pipeline.py build                      # extract, chunk, embed, index → index/
python rag_pipeline.py ask "What are the seven AI principles in Malaysia's AI Code of Ethics?"
python rag_pipeline.py batch                      # answer all 20 questions in questions.csv → results.jsonl
python evaluate_ragas.py                          # RAGAS scores → ragas_summary.csv, ragas_category_summary.csv
python threshold_analysis.py                      # threshold sweep → threshold_analysis.csv, threshold_plot.png
python threshold_analysis.py --refusal-as-escalation \
  --out-csv threshold_analysis_refusal_escalated.csv --out-png threshold_plot_refusal_escalated.png
```

To run the Opus comparison:

```bash
ANTHROPIC_MODEL=claude-opus-5-5 python rag_pipeline.py batch --out results_opus.jsonl
python evaluate_ragas.py --results results_opus.jsonl \
  --out-jsonl results_opus_with_ragas_scores.jsonl \
  --out-csv ragas_summary_opus.csv --out-cat-csv ragas_category_summary_opus.csv
python threshold_analysis.py --input results_opus_with_ragas_scores.jsonl \
  --out-csv threshold_analysis_opus.csv --out-png threshold_plot_opus.png
```

## Evaluation set

`questions.csv` has 20 questions, 5 in each category:

| Category | What it tests |
|---|---|
| Direct retrieval | The answer is stated in one passage |
| Conceptual | The answer must be synthesised within one document |
| Cross-document | The answer needs evidence from two or more documents |
| Unanswerable | The answer is not in the corpus, so the model should refuse |

An answer counts as **correct** when faithfulness ≥ 0.80, answer relevancy ≥ 0.80, and the model did not refuse. For unanswerable questions, a refusal counts as correct.

## Results

### Answer quality by category

| Category | Haiku 4.5 | Opus 5.5 |
|---|---:|---:|
| Direct retrieval | 4/5 (80%) | 5/5 (100%) |
| Conceptual | 4/5 (80%) | 5/5 (100%) |
| Cross-document | 3/5 (60%) | 3/5 (60%) |
| Unanswerable (correct refusal) | 5/5 (100%) | 5/5 (100%) |
| **All 20 questions** | **16/20 (80%)** | **18/20 (90%)** |

Average RAGAS scores over the 15 answerable questions:

| Metric | Haiku 4.5 | Opus 5.5 |
|---|---:|---:|
| Faithfulness | 0.88 | 0.97 |
| Answer relevancy | 0.84 | 0.79 |
| Context precision | 0.67 | 0.70 |
| Refusal rate | 13% | 0% |

Opus fixed Haiku's two faithfulness failures (Q04, Q08) and stopped refusing answerable questions. It still fails the two hardest cross-document questions (Q13, Q14). It no longer refuses them outright. It gives hedged partial answers ("the documents do not let me connect the two..."), which RAGAS scores at 0 for relevancy. Both models use the same retriever, so retrieval quality is identical. The difference comes entirely from generation.

### Confidence threshold (top-1 similarity)

At threshold **t = 0.80**:

| Setup | Auto-answer accuracy | Coverage | Escalated to human |
|---|---:|---:|---:|
| Haiku 4.5: refusal counts as a wrong answer | 71% (10/14) | 70% | 30% |
| Haiku 4.5: refusal counts as escalation | 83% (10/12) | 60% | 40% |
| Opus 5.5: refusal counts as a wrong answer | 86% (12/14) | 70% | 30% |

Full sweeps (0.55 to 0.85) are in `threshold_analysis*.csv` and `threshold_plot*.png`.

### Recommendation

Use **t = 0.80** and treat model refusals as escalations. With Haiku, this auto-answers 60% of questions at 83% accuracy and sends the other 40% to staff. Below 0.80 the threshold barely filters anything. At 0.85, coverage collapses to 10% (2 of 20 questions).

If accuracy matters more than cost, switching the generator to Opus 5.5 gives 86% accuracy at 70% coverage at the same threshold.

**Limitations:**
- No threshold reached the 95% accuracy target.
- Top-1 similarity scores are tightly clustered (category means range from 0.76 to 0.83), so the threshold is a weak confidence signal.
- With only 5 questions per category, one question changes a category's result by 20 percentage points.

## Known issues

From `error_analysis.csv`:

| Q | Category | Issue | Root cause | Proposed fix |
|---|---|---|---|---|
| Q13 | Cross-doc | Answerable question left unanswered | Vocabulary mismatch ("organisational AI governance" vs. "Pillar 3: Accountability") | Query decomposition: run one sub-query per document |
| Q14 | Cross-doc | Answerable question left unanswered | Stanford report (1,005 chunks) crowds out the Malaysia references | Metadata filtering by country, or map-reduce summarisation |
| Q08 | Conceptual | Low faithfulness (Haiku, 0.71) | Model added concepts that were not in the retrieved chunks | Stricter "context only" prompt; parent-document retrieval |
| Q04 | Direct | Low faithfulness (Haiku, 0.77) | 2021–2025 Roadmap mixed up with the 2026–2030 Action Plan | Tag documents by time period and filter on it |

## Files

| Path | Contents |
|---|---|
| `rag_pipeline.py` | Indexing, retrieval, and answer generation (`build` / `ask` / `batch`) |
| `evaluate_ragas.py` | RAGAS scoring |
| `threshold_analysis.py` | Confidence-threshold sweep and plot |
| `questions.csv` | 20 evaluation questions with reference answers |
| `data/` | Source PDFs |
| `index/` | FAISS index, chunks, and build metadata (`meta.json`) |
| `results*_with_ragas_scores.jsonl` | Answers, retrieved chunks, and scores for each question |
| `ragas_summary*.csv`, `ragas_category_summary*.csv` | Scores per question and per category |
| `threshold_analysis*.csv`, `threshold_plot*.png` | Threshold sweep results |
| `error_analysis.csv` | Failure analysis |

Files ending in `_opus` are from the Opus 5.5 run. Files without a suffix are from Haiku 4.5.
