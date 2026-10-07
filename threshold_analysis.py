"""
Confidence-threshold analysis for the RAG assistant.

Each answer is labelled correct when
    faithfulness >= 0.80 AND answer_relevancy >= 0.80 AND not refused.
top1_score (cosine similarity of the best retrieved chunk) is the confidence
signal: a query is auto-answered when top1_score >= threshold, otherwise it
is escalated to human staff.

For each threshold in 0.55..0.85 (step 0.05):
    accuracy        = correct auto-answered / auto-answered
    coverage        = auto-answered / all queries
    escalation_rate = escalated / all queries

Usage
    python threshold_analysis.py [--input results_with_ragas_scores.jsonl]
                                 [--target 0.95] [--refusal-as-escalation]

--refusal-as-escalation treats a model refusal as a hand-off to staff
(escalated, not a wrong auto-answer). Off by default, as per the spec.

Outputs: threshold_analysis.csv, threshold_plot.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

FAITH_MIN = 0.80
RELEVANCY_MIN = 0.80
THRESHOLDS = np.round(np.arange(0.55, 0.85 + 1e-9, 0.05), 2)

# Chart styling (light surface, single series)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


# ---------------------------------------------------------------------------
# Load + data checks
# ---------------------------------------------------------------------------

def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if "query_id" not in df:
        df["query_id"] = df.get("id", pd.Series(range(1, len(df) + 1))).astype(str)
    for col in ("faithfulness", "answer_relevancy", "top1_score"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["refused"] = df.get("refused", False).fillna(False).astype(bool)
    return df.sort_values("query_id").reset_index(drop=True)


def data_checks(df: pd.DataFrame) -> list[str]:
    issues = []
    dup = df[df["query_id"].duplicated()]["query_id"].tolist()
    if dup:
        issues.append(f"Duplicate query ids: {dup}")
    for col in ("faithfulness", "answer_relevancy", "top1_score"):
        missing = df[df[col].isna()]["query_id"].tolist()
        if missing:
            issues.append(f"Missing {col} (treated as not correct / never answered): {missing}")
    if "ragas_errors" in df:
        errs = df[df["ragas_errors"].apply(lambda e: bool(e))]["query_id"].tolist()
        if errs:
            issues.append(f"RAGAS scoring errors on: {errs}")
    if "error" in df:
        gen_err = df[df["error"].notna()]["query_id"].tolist()
        if gen_err:
            issues.append(f"Generation errors on: {gen_err}")

    if "answerable" in df:
        ans = df["answerable"].astype(bool)
        false_ref = df[ans & df["refused"]]["query_id"].tolist()
        if false_ref:
            issues.append(f"Answerable questions the model refused (false refusals): {false_ref}")
        correct_ref = df[~ans & df["refused"]]["query_id"].tolist()
        if correct_ref:
            issues.append(
                f"Unanswerable questions correctly refused but labelled WRONG by the "
                f"'refused = wrong' rule: {correct_ref}. Accuracy can never reach 100% at any "
                f"threshold that auto-answers them (see --refusal-as-escalation)."
            )
        halluc = df[~ans & ~df["refused"]]["query_id"].tolist()
        if halluc:
            issues.append(f"Unanswerable questions answered anyway (possible hallucination): {halluc}")

    # Refusals that carry extra text get non-trivial RAGAS scores.
    if "answer" in df:
        verbose = df[df["refused"] & (df["answer"].fillna("").str.len() > 80)]["query_id"].tolist()
        if verbose:
            issues.append(
                f"Refusals with extra explanation after the refusal sentence (RAGAS scored that "
                f"text, so their faithfulness is not meaningful): {verbose}"
            )
    if "answerable" in df:
        ref_scored = df[df["refused"] & (df["answer_relevancy"] == 0)]["query_id"].tolist()
        if ref_scored:
            issues.append(
                f"answer_relevancy = 0 on refusals {ref_scored}: RAGAS treats refusals as "
                f"noncommittal, so these are wrong by construction."
            )
    n = len(df)
    if n < 50:
        issues.append(
            f"Small sample (n={n}): each question moves coverage by {100 / n:.0f} points and "
            f"accuracy by more when few are answered - treat the curve as indicative only."
        )
    return issues


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def label_correct(df: pd.DataFrame) -> pd.Series:
    return (
        (df["faithfulness"] >= FAITH_MIN)
        & (df["answer_relevancy"] >= RELEVANCY_MIN)
        & ~df["refused"]
    ).fillna(False)


def sweep(df: pd.DataFrame, refusal_as_escalation: bool) -> pd.DataFrame:
    n = len(df)
    out = []
    for t in THRESHOLDS:
        confident = df["top1_score"] >= t  # NaN -> False -> escalated
        answered = confident & ~df["refused"] if refusal_as_escalation else confident
        n_ans = int(answered.sum())
        n_correct = int((answered & df["correct"]).sum())
        out.append({
            "threshold": float(t),
            "answered": n_ans,
            "escalated": n - n_ans,
            "correct_answered": n_correct,
            "wrong_answered": n_ans - n_correct,
            "accuracy": n_correct / n_ans if n_ans else np.nan,
            "coverage": n_ans / n,
            "escalation_rate": (n - n_ans) / n,
            "answered_ids": " ".join(df.loc[answered, "query_id"]),
            "wrong_ids": " ".join(df.loc[answered & ~df["correct"], "query_id"]),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(res: pd.DataFrame, target: float, path: Path, subtitle: str) -> None:
    pts = res.dropna(subset=["accuracy"])
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(pts["coverage"] * 100, pts["accuracy"] * 100, color=SERIES, lw=2, zorder=2)
    ax.scatter(pts["coverage"] * 100, pts["accuracy"] * 100, s=64, color=SERIES,
               edgecolor=SURFACE, linewidth=2, zorder=3)

    # One label per distinct point; thresholds that land on the same point are merged.
    for (cov, acc), grp in pts.groupby(["coverage", "accuracy"], sort=False):
        ts = grp["threshold"].tolist()
        label = f"t={ts[0]:.2f}" if len(ts) == 1 else f"t={ts[0]:.2f}–{ts[-1]:.2f}"
        ax.annotate(label, (cov * 100, acc * 100), textcoords="offset points", xytext=(8, 6),
                    fontsize=9, color=TEXT, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))

    ax.axhline(target * 100, color=TEXT_MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(1, target * 100 + 1, f"{target:.0%} accuracy target", fontsize=9, color=TEXT_MUTED,
            va="bottom")

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Coverage (% of queries auto-answered)", color=TEXT_MUTED)
    ax.set_ylabel("Accuracy of auto-answered queries (%)", color=TEXT_MUTED)
    ax.set_title("Accuracy vs coverage by top-1 retrieval threshold", loc="left",
                 fontsize=13, color=TEXT, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "results_with_ragas_scores.jsonl"))
    ap.add_argument("--target", type=float, default=0.95, help="accuracy target (default 0.95)")
    ap.add_argument("--refusal-as-escalation", action="store_true",
                    help="count model refusals as escalations instead of wrong answers")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    suffix = "_refusal_escalated" if args.refusal_as_escalation else ""
    out_csv = Path(args.out_csv or ROOT / f"threshold_analysis{suffix}.csv")
    out_png = Path(args.out_png or ROOT / f"threshold_plot{suffix}.png")

    df = load(Path(args.input))
    df["correct"] = label_correct(df)

    print(f"Loaded {len(df)} answers from {args.input}")
    if "served_by" in df:
        print(f"Generator: {', '.join(sorted(df['served_by'].dropna().unique()))}"
              + (f" | Judge: {', '.join(sorted(df['judge_model'].dropna().unique()))}" if "judge_model" in df else ""))
    print(f"Correct = faithfulness >= {FAITH_MIN} AND answer_relevancy >= {RELEVANCY_MIN} AND not refused"
          f"{' (refusals counted as escalations)' if args.refusal_as_escalation else ''}")
    print(f"Correct answers: {int(df['correct'].sum())}/{len(df)}")

    # Per-question view
    cols = ["query_id", "top1_score", "faithfulness", "answer_relevancy", "refused", "correct"]
    print("\nPer-question labels")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    c, w = df.loc[df["correct"], "top1_score"], df.loc[~df["correct"], "top1_score"]
    print(f"\ntop1_score  correct: {c.min():.3f}–{c.max():.3f} (mean {c.mean():.3f}) | "
          f"not correct: {w.min():.3f}–{w.max():.3f} (mean {w.mean():.3f})")

    issues = data_checks(df)
    print("\nData / extraction issues")
    for i in issues or ["None found."]:
        print(f"  - {i}")

    res = sweep(df, args.refusal_as_escalation)
    csv_out = res.copy()
    for col in ("accuracy", "coverage", "escalation_rate"):
        csv_out[col] = csv_out[col].round(4)
    csv_out.to_csv(out_csv, index=False)

    subtitle = (f"n={len(df)} queries · correct = faithfulness ≥ {FAITH_MIN:.2f}, relevancy ≥ "
                f"{RELEVANCY_MIN:.2f}, not refused"
                + (" · refusals escalated" if args.refusal_as_escalation else ""))
    plot(res, args.target, out_png, subtitle)

    # Full table
    show = res.drop(columns=["answered_ids"]).copy()
    for col in ("accuracy", "coverage", "escalation_rate"):
        show[col] = show[col].map(lambda x: "  n/a" if pd.isna(x) else f"{x:6.1%}")
    print(f"\nThreshold sweep ({out_csv.name})")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Recommendation: the lowest threshold (= highest coverage) meeting the target.
    ok = res[res["accuracy"] >= args.target - 1e-9]
    print("\nRecommendation")
    if not ok.empty:
        best = ok.sort_values(["coverage", "threshold"], ascending=[False, True]).iloc[0]
        print(f"  To achieve {args.target:.0%} accuracy, threshold {best['threshold']:.2f} gives "
              f"{best['coverage']:.0%} coverage and {best['escalation_rate']:.0%} escalation "
              f"({best['answered']} answered, {best['escalated']} escalated).")
    else:
        valid = res.dropna(subset=["accuracy"])
        best = valid.sort_values(["accuracy", "coverage"], ascending=[False, False]).iloc[0]
        print(f"  No threshold in {THRESHOLDS[0]:.2f}–{THRESHOLDS[-1]:.2f} reaches {args.target:.0%} accuracy.")
        print(f"  Best achievable: threshold {best['threshold']:.2f} gives {best['accuracy']:.0%} accuracy, "
              f"{best['coverage']:.0%} coverage and {best['escalation_rate']:.0%} escalation "
              f"({best['answered']} answered, {best['escalated']} escalated).")
        print("  top1_score alone does not separate correct from incorrect answers here - "
              "their score ranges overlap (see above).")

    print(f"\nWrote {out_csv.name} and {out_png.name}")


if __name__ == "__main__":
    main()
