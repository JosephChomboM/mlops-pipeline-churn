"""Markdown summary of a simulation run."""

import json

from churn import config


def _money(x: float) -> str:
    return f"{'-' if x < 0 else '+'}${abs(x):,.0f}"


def _month_row(r: dict) -> str:
    validation = "quarantined" if "validation" in r else "ok"
    drift = r.get("drift")
    psi = f"{drift['max_feature_psi']:.3f} ({drift['top_feature']})" if drift else "-"
    pred_psi = f"{drift['prediction_psi']:.3f}" if drift else "-"
    realized = r.get("realized")
    if realized:
        low, high = realized["gini_ci95"]
        gini = (
            f"{realized['gini']:.3f} [{low:.2f}-{high:.2f}] ({realized['snapshot_date']}, v{realized['model_version']})"
        )
        calib = f"{realized['predicted_churn_rate']:.1%} / {realized['churn_rate']:.1%}"
    else:
        gini, calib = "pending", "-"
    action = r["retraining"]["action"]
    if "challenger_version" in r["retraining"]:
        action += f" v{r['retraining']['challenger_version']}"
    champion = f"v{r['champion_before']}"
    if r["champion_after"] != r["champion_before"]:
        champion += f" -> v{r['champion_after']}"
    signals = ", ".join(r["signals"]) or "-"
    return f"| {r['as_of']} | {r['shift']} | {validation} | {psi} | {pred_psi} | {gini} | {calib} | {signals} | {action} | {champion} |"


def write_summary(records: list[dict]) -> None:
    lines = [
        "# Simulated production: six months",
        "",
        "| Month | Shift | Validation | Max feature PSI | Prediction PSI | Realized Gini [95% CI] (matured cohort) "
        "| Predicted / actual churn | Signals | Action | Champion |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[_month_row(r) for r in records],
        "",
        "## Promotion decisions (frozen holdout)",
        "",
        "| Challenger | Gini | KS | ECE | Net value | vs champion | Difference [95% CI] | Wins | Result |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for line in (config.REPORTS_DIR / "promotions.jsonl").read_text().splitlines():
        p = json.loads(line)
        ch, cp, boot = p["challenger"], p["champion"], p.get("bootstrap")
        vs = f"v{cp['version']}: Gini {cp['gini']}, ${cp['net_value']:,.0f}" if cp else "-"
        if boot:
            low, high = (_money(x) for x in boot["ci95"])
            diff = f"{_money(boot['observed_diff'])} [{low}, {high}]"
            wins = f"{boot['win_probability']:.0%}"
        else:
            diff, wins = "-", "-"
        result = "promoted" if p["promoted"] else "rejected"
        lines.append(
            f"| v{ch['version']} | {ch['gini']} | {ch['ks']} | {ch['ece']} | ${ch['net_value']:,.0f} | {vs} "
            f"| {diff} | {wins} | {result} |"
        )
    lines += [
        "",
        f"A challenger is promoted only if it wins at least {config.PROMOTION_MIN_WIN_PROBABILITY:.0%} of "
        f"{config.PROMOTION_BOOTSTRAP_SAMPLES:,} paired bootstrap resamples of the holdout.",
    ]
    (config.REPORTS_DIR / "simulation.md").write_text("\n".join(lines) + "\n")
