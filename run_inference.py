"""
run_inference.py
----------------
Run the trained (or baseline) triage agent on new support tickets in real-time.

This script demonstrates the PRODUCTION use of the agent:
  - Load the optimized prompt from training (or use baseline)
  - Process any new ticket interactively or from a list
  - Display the triage result with category, priority, team, and reasoning

Run:
    python run_inference.py

This does NOT require agentlightning at inference time — it directly
calls Azure OpenAI GPT-4.1 with the optimized prompt.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from agent.grader import compute_reward
from agent.triage_agent import BASELINE_PROMPT, TRIAGE_TOOL, DEPLOYMENT, get_azure_client

load_dotenv()

# ─── Priority / urgency colors for CLI output ──────────────────────────────────

PRIORITY_EMOJI = {
    "P1-Critical": "🔴",
    "P2-High":     "🟠",
    "P3-Medium":   "🟡",
    "P4-Low":      "🟢",
}

TEAM_EMOJI = {
    "SRE":          "⚙️ ",
    "Tier2":        "🔧",
    "Billing":      "💳",
    "Account-Mgmt": "👔",
    "Tier1":        "💬",
}

# ─── Load prompt ──────────────────────────────────────────────────────────────

def load_prompt() -> str:
    """Load optimized prompt if available, otherwise use baseline."""
    prompt_file = Path("optimized_prompt.txt")
    if prompt_file.exists():
        prompt = prompt_file.read_text().strip()
        print("✅ Loaded optimized prompt from optimized_prompt.txt")
        return prompt
    else:
        print("⚠️  No optimized prompt found. Using baseline prompt.")
        print("   Run train_apo.py first to optimize the agent.")
        return BASELINE_PROMPT


# ─── Core inference function ──────────────────────────────────────────────────

def triage_ticket(
    ticket_text: str,
    customer_tier: str,
    prompt_template: str,
    client: OpenAI,
) -> dict:
    """
    Triage a single support ticket using the trained agent.

    Args:
        ticket_text: Raw ticket content from the customer
        customer_tier: Free | Pro | Enterprise
        prompt_template: The (possibly APO-optimized) system prompt
        client: Azure OpenAI client

    Returns:
        dict with category, priority, team, reasoning
    """
    user_prompt = prompt_template.format(
        customer_tier=customer_tier,
        ticket_text=ticket_text,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise customer support triage agent. "
                "Always use the submit_triage function to record your decision."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        tools=[TRIAGE_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_triage"}},
        temperature=0.1,
        max_tokens=512,
    )

    tool_calls = response.choices[0].message.tool_calls
    result = {
        "category": "General",
        "priority": "P4-Low",
        "team": "Tier1",
        "reasoning": "No classification returned.",
    }

    if tool_calls:
        try:
            args = json.loads(tool_calls[0].function.arguments)
            result.update(args)
        except (json.JSONDecodeError, KeyError):
            pass

    return result


# ─── Pretty print result ──────────────────────────────────────────────────────

def print_result(ticket_text: str, customer_tier: str, result: dict, expected: dict | None = None):
    """Print a formatted triage result to the console."""
    p_emoji = PRIORITY_EMOJI.get(result["priority"], "⚪")
    t_emoji = TEAM_EMOJI.get(result["team"], "🔹")

    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print(f"│  TICKET TRIAGE RESULT                     [{customer_tier:^12}]  │")
    print("├─────────────────────────────────────────────────────────────┤")
    # Wrap ticket text at 60 chars
    words = ticket_text.split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) + 1 <= 57:
            line = (line + " " + word).strip()
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    for ln in lines[:3]:  # Max 3 lines preview
        print(f"│  {ln:<59}  │")
    if len(lines) > 3:
        print(f"│  {'...':<59}  │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  Category : {result['category']:<49}  │")
    print(f"│  Priority : {p_emoji} {result['priority']:<47}  │")
    print(f"│  Route To : {t_emoji} {result['team']:<47}  │")
    print("├─────────────────────────────────────────────────────────────┤")
    # Wrap reasoning
    reasoning_words = result.get("reasoning", "").split()
    r_lines, r_line = [], ""
    for word in reasoning_words:
        if len(r_line) + len(word) + 1 <= 57:
            r_line = (r_line + " " + word).strip()
        else:
            r_lines.append(r_line)
            r_line = word
    if r_line:
        r_lines.append(r_line)
    for ln in r_lines[:3]:
        print(f"│  {ln:<59}  │")
    print("└─────────────────────────────────────────────────────────────┘")

    if expected:
        reward = compute_reward(
            predicted_category=result["category"],
            predicted_priority=result["priority"],
            predicted_team=result["team"],
            expected_category=expected["category"],
            expected_priority=expected["priority"],
            expected_team=expected["team"],
        )
        correct = (
            result["category"] == expected["category"] and
            result["priority"] == expected["priority"] and
            result["team"] == expected["team"]
        )
        status = "✅ CORRECT" if correct else "⚠️  PARTIAL"
        print(f"  Score: {reward:.2f}/1.00  {status}  "
              f"(Expected: {expected['category']} | {expected['priority']} | {expected['team']})")


# ─── Demo tickets ─────────────────────────────────────────────────────────────

DEMO_TICKETS = [
    {
        "text": "CRITICAL: Our entire payment processing pipeline is down. Zero transactions going through since 2 AM. Losing $80K per hour.",
        "customer_tier": "Enterprise",
        "expected": {"category": "Technical", "priority": "P1-Critical", "team": "SRE"},
    },
    {
        "text": "We were charged twice for our Pro subscription this month. Please refund the duplicate charge of $299.",
        "customer_tier": "Pro",
        "expected": {"category": "Billing", "priority": "P2-High", "team": "Billing"},
    },
    {
        "text": "The search functionality returns wrong results when using special characters like & and %. Other characters work fine.",
        "customer_tier": "Pro",
        "expected": {"category": "Technical", "priority": "P2-High", "team": "Tier2"},
    },
    {
        "text": "Hi! I'm trying to connect my Google Sheets. Do you have a tutorial or step-by-step guide?",
        "customer_tier": "Free",
        "expected": {"category": "General", "priority": "P4-Low", "team": "Tier1"},
    },
    {
        "text": "We need to add 10 new seats to our Enterprise plan and update billing to our new CFO. Also need a new invoice.",
        "customer_tier": "Enterprise",
        "expected": {"category": "Account", "priority": "P3-Medium", "team": "Account-Mgmt"},
    },
]


def main():
    print("=" * 65)
    print("  Customer Support Triage Agent — Real-Time Inference")
    print("  Powered by Azure OpenAI GPT-4.1")
    print("=" * 65)
    print()

    prompt = load_prompt()
    client = get_azure_client()

    print()
    print(f"🎯 Processing {len(DEMO_TICKETS)} demo tickets...")
    print()

    scores = []
    for i, demo in enumerate(DEMO_TICKETS, 1):
        print(f"[Ticket {i}/{len(DEMO_TICKETS)}] Triaging...")
        result = triage_ticket(
            ticket_text=demo["text"],
            customer_tier=demo["customer_tier"],
            prompt_template=prompt,
            client=client,
        )
        print_result(demo["text"], demo["customer_tier"], result, demo["expected"])
        scores.append(
            compute_reward(
                result["category"], result["priority"], result["team"],
                demo["expected"]["category"], demo["expected"]["priority"], demo["expected"]["team"],
            )
        )

    avg_score = sum(scores) / len(scores)
    print()
    print("=" * 65)
    print(f"  📊 Overall Score: {avg_score:.2f}/1.00  ({avg_score*100:.1f}%)")
    print("=" * 65)
    print()
    print("💡 To improve accuracy, run: python train_apo.py")
    print()

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("─" * 65)
    print("  🖊️  Interactive Mode — Triage your own ticket")
    print("─" * 65)
    print("  (Press Ctrl+C to exit)")
    print()

    while True:
        try:
            print("Enter ticket text (or 'quit' to exit):")
            ticket_text = input("  > ").strip()
            if ticket_text.lower() in ("quit", "exit", "q"):
                break
            if not ticket_text:
                continue

            print("Customer tier (Free/Pro/Enterprise) [default: Free]:")
            tier = input("  > ").strip() or "Free"
            if tier not in ("Free", "Pro", "Enterprise"):
                tier = "Free"

            print("\n⏳ Triaging...")
            result = triage_ticket(ticket_text, tier, prompt, client)
            print_result(ticket_text, tier, result)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
