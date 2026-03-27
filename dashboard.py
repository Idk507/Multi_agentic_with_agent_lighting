"""
dashboard.py
------------
CLI dashboard to inspect training results and compare baseline vs optimized agent.

Run AFTER train_apo.py has completed:
    python dashboard.py

Shows:
  - Baseline prompt vs optimized prompt (diff)
  - Per-ticket accuracy breakdown across val dataset
  - Category / Priority / Team accuracy separately
  - Confusion matrix for priority classification
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from agent.triage_agent import BASELINE_PROMPT, TRIAGE_TOOL, DEPLOYMENT, get_azure_client
from agent.grader import compute_reward
from data.tickets_dataset import get_val_dataset

PRIORITY_ORDER = ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"]
CATEGORIES = ["Technical", "Billing", "Account", "General"]
TEAMS = ["SRE", "Tier2", "Billing", "Account-Mgmt", "Tier1"]

PRIORITY_EMOJI = {
    "P1-Critical": "🔴",
    "P2-High":     "🟠",
    "P3-Medium":   "🟡",
    "P4-Low":      "🟢",
}


def run_agent_on_dataset(prompt: str, dataset: list, label: str) -> list[dict]:
    """Run the triage agent on the full validation dataset and return results."""
    client = get_azure_client()
    results = []

    print(f"\n  Running {label} on {len(dataset)} validation tickets...")

    for i, task in enumerate(dataset):
        user_prompt = prompt.format(
            customer_tier=task["customer_tier"],
            ticket_text=task["text"],
        )

        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a customer support triage agent. Always use the submit_triage function."},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[TRIAGE_TOOL],
                tool_choice={"type": "function", "function": {"name": "submit_triage"}},
                temperature=0.1,
                max_tokens=512,
            )

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                args = json.loads(tool_calls[0].function.arguments)
                pred_cat  = args.get("category", "General")
                pred_pri  = args.get("priority", "P4-Low")
                pred_team = args.get("team", "Tier1")
            else:
                pred_cat, pred_pri, pred_team = "General", "P4-Low", "Tier1"

        except Exception as e:
            print(f"    ⚠️  Ticket {task['ticket_id']} failed: {e}")
            pred_cat, pred_pri, pred_team = "General", "P4-Low", "Tier1"

        reward = compute_reward(
            pred_cat, pred_pri, pred_team,
            task["expected_category"], task["expected_priority"], task["expected_team"],
        )

        results.append({
            "ticket_id":  task["ticket_id"],
            "text":       task["text"][:60] + "...",
            "tier":       task["customer_tier"],
            "pred_cat":   pred_cat,
            "pred_pri":   pred_pri,
            "pred_team":  pred_team,
            "exp_cat":    task["expected_category"],
            "exp_pri":    task["expected_priority"],
            "exp_team":   task["expected_team"],
            "reward":     reward,
        })

        status = "✅" if reward == 1.0 else ("⚠️ " if reward > 0.5 else "❌")
        print(f"    {status} [{task['ticket_id']}] reward={reward:.2f} | "
              f"pred={pred_cat}/{pred_pri}/{pred_team}")

    return results


def print_metrics(results: list[dict], label: str):
    """Print detailed metrics for a set of results."""
    total = len(results)
    avg_reward = sum(r["reward"] for r in results) / total
    perfect = sum(1 for r in results if r["reward"] == 1.0)

    cat_correct  = sum(1 for r in results if r["pred_cat"]  == r["exp_cat"])
    pri_correct  = sum(1 for r in results if r["pred_pri"]  == r["exp_pri"])
    team_correct = sum(1 for r in results if r["pred_team"] == r["exp_team"])

    print(f"\n  ── {label} ──")
    print(f"  Avg Reward   : {avg_reward:.3f} / 1.000   ({avg_reward*100:.1f}%)")
    print(f"  Perfect (1.0): {perfect}/{total}   ({perfect/total*100:.1f}%)")
    print(f"  Category Acc : {cat_correct}/{total}   ({cat_correct/total*100:.1f}%)")
    print(f"  Priority Acc : {pri_correct}/{total}   ({pri_correct/total*100:.1f}%)")
    print(f"  Team Acc     : {team_correct}/{total}   ({team_correct/total*100:.1f}%)")

    return avg_reward


def print_per_ticket_table(results: list[dict]):
    """Print per-ticket breakdown table."""
    print(f"\n  {'ID':<6} {'Tier':<12} {'Cat':<11} {'Priority':<14} {'Team':<13} {'Score'}")
    print(f"  {'─'*6} {'─'*12} {'─'*11} {'─'*14} {'─'*13} {'─'*5}")
    for r in results:
        p_ok = "✓" if r["pred_pri"]  == r["exp_pri"]  else "✗"
        c_ok = "✓" if r["pred_cat"]  == r["exp_cat"]  else "✗"
        t_ok = "✓" if r["pred_team"] == r["exp_team"] else "✗"
        emoji = PRIORITY_EMOJI.get(r["pred_pri"], "⚪")
        print(
            f"  {r['ticket_id']:<6} {r['tier']:<12} "
            f"{c_ok}{r['pred_cat']:<10} "
            f"{p_ok}{emoji}{r['pred_pri']:<12} "
            f"{t_ok}{r['pred_team']:<12} "
            f"{r['reward']:.2f}"
        )


def print_confusion(results: list[dict], field_pred: str, field_exp: str, labels: list[str], title: str):
    """Print a simple confusion matrix."""
    matrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        matrix[r[field_exp]][r[field_pred]] += 1

    print(f"\n  {title} Confusion Matrix (rows=actual, cols=predicted):")
    col_w = 10
    header = f"  {'Actual':<14}" + "".join(f"{l[:col_w-1]:<{col_w}}" for l in labels)
    print(header)
    print("  " + "─" * (14 + col_w * len(labels)))
    for actual in labels:
        row = f"  {actual:<14}"
        for predicted in labels:
            count = matrix[actual][predicted]
            marker = f"[{count}]" if actual == predicted else f" {count} "
            row += f"{marker:<{col_w}}"
        print(row)


def main():
    print("=" * 65)
    print("  Customer Support Triage Agent — Training Dashboard")
    print("=" * 65)

    val_dataset = get_val_dataset()

    # Load prompts
    baseline_prompt = BASELINE_PROMPT

    optimized_prompt_path = Path("optimized_prompt.txt")
    has_optimized = optimized_prompt_path.exists()

    if has_optimized:
        optimized_prompt = optimized_prompt_path.read_text().strip()
        print(f"\n  ✅ Optimized prompt found at optimized_prompt.txt")
        print(f"  📏 Baseline length : {len(baseline_prompt)} chars")
        print(f"  📏 Optimized length: {len(optimized_prompt)} chars")
    else:
        print(f"\n  ⚠️  No optimized prompt found. Run train_apo.py first.")
        print(f"     Showing baseline performance only.")
        optimized_prompt = None

    print(f"\n{'─'*65}")
    print("  EVALUATION — Validation Dataset")
    print(f"{'─'*65}")

    # Run baseline
    baseline_results = run_agent_on_dataset(baseline_prompt, val_dataset, "Baseline Prompt")

    if has_optimized:
        optimized_results = run_agent_on_dataset(optimized_prompt, val_dataset, "Optimized Prompt (APO)")

    print(f"\n{'─'*65}")
    print("  METRICS COMPARISON")
    print(f"{'─'*65}")

    baseline_score = print_metrics(baseline_results, "BASELINE")

    if has_optimized:
        optimized_score = print_metrics(optimized_results, "OPTIMIZED (APO)")
        improvement = (optimized_score - baseline_score) * 100
        direction = "📈" if improvement > 0 else "📉"
        print(f"\n  {direction} Improvement : {improvement:+.1f} percentage points")

    print(f"\n{'─'*65}")
    print("  PER-TICKET BREAKDOWN — Baseline")
    print(f"{'─'*65}")
    print_per_ticket_table(baseline_results)

    if has_optimized:
        print(f"\n{'─'*65}")
        print("  PER-TICKET BREAKDOWN — Optimized")
        print(f"{'─'*65}")
        print_per_ticket_table(optimized_results)

    print(f"\n{'─'*65}")
    print("  CONFUSION MATRICES — Baseline")
    print(f"{'─'*65}")
    print_confusion(baseline_results, "pred_pri", "exp_pri", PRIORITY_ORDER, "Priority")
    print_confusion(baseline_results, "pred_team", "exp_team", TEAMS, "Team")

    if has_optimized:
        print(f"\n{'─'*65}")
        print("  CONFUSION MATRICES — Optimized")
        print(f"{'─'*65}")
        print_confusion(optimized_results, "pred_pri", "exp_pri", PRIORITY_ORDER, "Priority")
        print_confusion(optimized_results, "pred_team", "exp_team", TEAMS, "Team")

    print()


if __name__ == "__main__":
    main()
