# Terminal Evidence for Report

Selected outputs from the terminal runs. Everything else is already in the CSV/JSONL files. Use these as appendix material, screenshots for the video, or short quotes in the report.

---

## 1. Run configuration (as shown in terminal)

| Item | Haiku run | Opus run |
|---|---|---|
| Generator | claude-haiku-4-5-20251001 | claude-opus-5-5 |
| Temperature / effort | 0.0 / none | none / medium |
| Judge (final scores) | claude-sonnet-5-5 | claude-sonnet-5-5 |
| Retrieval | bge-small-en-v1.5, top_k=5 | same |
| Questions | 20 | 20 |
| Refused in batch run | 7 (Q13, Q14, Q16-Q20) | 5 (Q16-Q20) |
| Judge calls | 200 | 200 |
| Judge tokens in / out | 386,671 / 44,136 | 406,063 / 55,744 |

Single-question cost comparison (same question, "key principles of Malaysia's AI Code of Ethics"):

| Model | Tokens in/out | Latency |
|---|---|---|
| Haiku 4.5 | 1,245 / 145 | 1.71 s |
| Opus 5.5 | 1,875 / 411 | 6.25 s |

Use for: the cost / latency trade-off between generators.

---

## 2. Sample answer with citations (good for the appendix or video demo)

```
Q: What are the key principles of Malaysia's AI Code of Ethics?
--------------------------------------------------------------------------------
According to Malaysia's AI Code of Ethics, there are Seven (7) key principles [2]:

1. Fairness [2]
2. Reliability, Safety and Control [2]
3. Privacy and Security [2]
4. Inclusiveness [2]
5. Transparency [2]
6. Accountability [2]
7. Pursuit of human benefit and happiness [2]

These principles are outlined in the National Guidelines on AI Governance and Ethics (AIGE), and the AI Code of Ethics (AICE) translates them into actionable measures to ensure they are embedded throughout the entire lifecycle of an AI system [2].
--------------------------------------------------------------------------------
Sources:
  [1] AI_Code_of_Ethics_Malaysia.pdf p.1  (cosine=0.8459)
  [2] AI_Code_of_Ethics_Malaysia.pdf p.4  (cosine=0.8418)
  [3] AI_Code_of_Ethics_Malaysia.pdf p.3  (cosine=0.7906)
  [4] AI_Boardroom_Malaysia.pdf p.28  (cosine=0.7686)
  [5] AI_Boardroom_Malaysia.pdf p.2  (cosine=0.7607)
top1_score=0.8459  mean_topk_score=0.8015  refused=False
```

Use for: showing inline citations [n], source file + page, and cosine scores.

---

## 3. False refusals (Haiku) - evidence for the error analysis

**Q13** - top1 = 0.8500, all 5 retrieved chunks from `AI_Boardroom_Malaysia.pdf` (pp. 1, 4, 6, 2, 5). No national-policy document retrieved.

> "I cannot answer this question from the provided documents. The context passages describe what the Boardroom Primer contains and its purpose [3], but they do not provide information about Malaysia's national approach to AI governance that would allow for a comparison."

**Q14** - top1 = 0.7612. The only Stanford chunk retrieved was p.2, the table of contents.

> "I cannot answer this question from the provided documents. ... The passages from the Stanford report only show a table of contents, and there is no substantive content from that report that discusses Malaysia."

Use for: proving the failure is retrieval (wrong or missing chunks), not generation. The model refused correctly given what it was shown.

---

## 4. RAGAS results - Haiku 4.5 generator (judge: Sonnet 5.5)

| id | category | faith | relev | ctxP | rel/5 | top1 | outcome |
|---|---|---|---|---|---|---|---|
| Q01 | direct | 1.000 | 0.997 | 1.000 | 1 | 0.831 | answered |
| Q02 | direct | 1.000 | 1.000 | 0.583 | 2 | 0.863 | answered |
| Q03 | direct | 1.000 | 1.000 | 0.750 | 2 | 0.829 | answered |
| Q04 | direct | 0.773 | 0.908 | 0.639 | 3 | 0.803 | answered |
| Q05 | direct | 1.000 | 0.943 | 1.000 | 1 | 0.804 | answered |
| Q06 | conceptual | 1.000 | 1.000 | 1.000 | 4 | 0.839 | answered |
| Q07 | conceptual | 0.938 | 0.881 | 0.333 | 1 | 0.809 | answered |
| Q08 | conceptual | 0.714 | 1.000 | 0.333 | 1 | 0.846 | answered |
| Q09 | conceptual | 0.909 | 0.962 | 0.639 | 3 | 0.813 | answered |
| Q10 | conceptual | 0.889 | 0.997 | 0.367 | 2 | 0.801 | answered |
| Q11 | cross_doc | 0.818 | 0.981 | 0.833 | 2 | 0.819 | answered |
| Q12 | cross_doc | 1.000 | 1.000 | 1.000 | 5 | 0.810 | answered |
| Q13 | cross_doc | 0.571 | 0.000 | 0.367 | 2 | 0.850 | false_refusal |
| Q14 | cross_doc | 0.714 | 0.000 | 0.200 | 1 | 0.761 | false_refusal |
| Q15 | cross_doc | 0.846 | 0.972 | 1.000 | 5 | 0.774 | answered |
| Q16 | unanswerable | 1.000 | 0.000 | 0.450 | 2 | 0.780 | correct_refusal |
| Q17 | unanswerable | 1.000 | 0.000 | 0.000 | 0 | 0.745 | correct_refusal |
| Q18 | unanswerable | 1.000 | 0.000 | 0.000 | 0 | 0.681 | correct_refusal |
| Q19 | unanswerable | 1.000 | 0.000 | 1.000 | 1 | 0.779 | correct_refusal |
| Q20 | unanswerable | 1.000 | 0.000 | 0.867 | 3 | 0.803 | correct_refusal |

By category:

| category | n | faith | relev | ctxP | rel/5 | top1 | refused |
|---|---|---|---|---|---|---|---|
| direct_retrieval | 5 | 0.955 | 0.970 | 0.794 | 1.80 | 0.826 | 0% |
| conceptual | 5 | 0.890 | 0.968 | 0.534 | 2.20 | 0.822 | 0% |
| cross_document | 5 | 0.790 | 0.591 | 0.680 | 3.00 | 0.803 | 40% |
| unanswerable | 5 | 1.000 | 0.000 | 0.463 | 1.20 | 0.758 | 100% |
| ALL_ANSWERABLE | 15 | 0.878 | 0.843 | 0.670 | 2.33 | 0.817 | 13% |

---

## 5. RAGAS results - Opus 5.5 generator (judge: Sonnet 5.5)

| category | n | faith | relev | ctxP | rel/5 | top1 | refused |
|---|---|---|---|---|---|---|---|
| direct_retrieval | 5 | 0.983 | 0.903 | 0.878 | 2.00 | 0.826 | 0% |
| conceptual | 5 | 0.958 | 0.907 | 0.534 | 2.20 | 0.822 | 0% |
| cross_document | 5 | 0.957 | 0.559 | 0.680 | 3.00 | 0.803 | 0% |
| unanswerable | 5 | 1.000 | 0.000 | 0.251 | 0.80 | 0.758 | 100% |
| ALL_ANSWERABLE | 15 | 0.966 | 0.790 | 0.697 | 2.40 | 0.817 | 0% |

Per-question highlights (Opus):
- Q04 faithfulness 0.773 -> 1.000 and Q08 0.714 -> 0.889: a stronger generator fixed the grounding failures.
- Q13 and Q14 still have answer relevancy = 0.000: the generator cannot fix a retrieval failure.
- Faithfulness of Q13 rose (0.571 -> 1.000) only because Opus hedged ("documents do not cover this") instead of refusing, so the model did not invent anything. It still failed to answer.
- Opus marked 9 answerable questions as `answered_with_caveat` (Q04, Q06, Q08, Q10-Q15). It states what the passages do not cover.

---

## 6. Threshold analysis output (Opus run)

Rule: correct = faithfulness >= 0.8 AND answer_relevancy >= 0.8 AND not refused. Correct answers: 13/20.

Score separation: correct answers had top1 0.774-0.863 (mean 0.819); incorrect had 0.681-0.850 (mean 0.771). The ranges overlap.

| threshold | answered | escalated | correct | wrong | accuracy | coverage | escalation | wrong ids |
|---|---|---|---|---|---|---|---|---|
| 0.55 | 20 | 0 | 13 | 7 | 65.0% | 100% | 0% | Q13 Q14 Q16-Q20 |
| 0.60 | 20 | 0 | 13 | 7 | 65.0% | 100% | 0% | Q13 Q14 Q16-Q20 |
| 0.65 | 20 | 0 | 13 | 7 | 65.0% | 100% | 0% | Q13 Q14 Q16-Q20 |
| 0.70 | 19 | 1 | 13 | 6 | 68.4% | 95% | 5% | Q13 Q14 Q16 Q17 Q19 Q20 |
| 0.75 | 18 | 2 | 13 | 5 | 72.2% | 90% | 10% | Q13 Q14 Q16 Q19 Q20 |
| 0.80 | 14 | 6 | 12 | 2 | 85.7% | 70% | 30% | Q13 Q20 |
| 0.85 | 2 | 18 | 1 | 1 | 50.0% | 10% | 90% | Q13 |

Recommendation printed by the script:
- No threshold in 0.55-0.85 reaches the 95% accuracy target.
- Best achievable: **t = 0.80 -> 85.7% accuracy, 70% coverage, 30% escalation** (14 answered, 6 escalated).
- top1_score alone does not separate correct from incorrect answers.

Data issues flagged by the script (put these in the limitations section):
1. The five unanswerable questions (Q16-Q20) were correctly refused, but the "refused = wrong" rule labels them wrong, so accuracy cannot reach 100% unless refusals are treated as escalations.
2. RAGAS gives answer_relevancy = 0 on refusals (treated as noncommittal), so those rows are wrong by construction.
3. n = 20: each question moves coverage by 5 points, so the curve is indicative only.

Key insight for the report: retrieval score alone is a weak confidence signal. Q13 had the **highest** top1 (0.850) and still failed. A better signal would combine the retrieval score with source diversity or an LLM self-check.

---

## 7. Implementation challenges (from the terminal)

| Problem | What happened | Fix |
|---|---|---|
| RAGAS crash | `TypeError: float has no len()` in `evaluate_ragas.py` (line 208, faithfulness result object) when scoring all 20 answers | Patched the score handling in the script |
| Segmentation fault | `zsh: segmentation fault python evaluate_ragas.py` with a leaked loky semaphore warning, during the first full run | Re-ran after the fix and killed stuck runs |
| Hung run | Ctrl+C during async embedding step (answer relevancy) left threads waiting | Tested with `--only Q01,Q16`, then re-ran the full set |
| Judge choice | Early runs used claude-opus-5-5 as judge | Final runs for both generators use claude-sonnet-5-5 so the comparison is consistent |
| HF Hub warning | Unauthenticated requests warning on every run | Harmless (model is cached locally); optionally set `HF_TOKEN` |

---

## 8. What NOT to paste into the report

- Raw tracebacks and the "Loading weights" progress bars.
- The repeated `[1/20] top1=... refused=...` batch lines. Summarise them (5-7 refusals per run) instead.
- Full answer texts for all 20 questions. They are in `results.jsonl` and the Opus JSONL.
