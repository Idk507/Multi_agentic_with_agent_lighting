"""
test_connection.py
------------------
Validates your setup BEFORE running the full APO training.

Tests:
  1. ✅ .env loaded correctly
  2. ✅ Azure OpenAI connection works (GPT-4.1 live call)
  3. ✅ Grader reward function is correct
  4. ✅ Agent tool-call flow works end-to-end on one ticket

Run:
    python test_connection.py

All tests should pass before you run train_apo.py
"""

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

PASS = "✅"
FAIL = "❌"
SEP  = "─" * 55


def test_env():
    """Test 1: Check all required environment variables are set."""
    print(f"\n{SEP}")
    print("TEST 1: Environment Variables")
    print(SEP)

    required_vars = [
        "AI_FOUNDRY_PROJECT_ENDPOINT",
        "AI_FOUNDRY_DEPLOYMENT_NAME",
        "AI_FOUNDRY_API_VERSION",
        "AI_FOUNDRY_API_KEY",
    ]

    all_ok = True
    for var in required_vars:
        val = os.environ.get(var, "")
        if val:
            # Mask the API key for security
            display = val[:10] + "..." if "KEY" in var else val
            print(f"  {PASS}  {var} = {display}")
        else:
            print(f"  {FAIL}  {var} is NOT SET")
            all_ok = False

    return all_ok


def test_azure_connection():
    """Test 2: Make a live call to Azure OpenAI."""
    print(f"\n{SEP}")
    print("TEST 2: Azure OpenAI Connection (Live Call)")
    print(SEP)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["AI_FOUNDRY_API_KEY"],
            base_url=(
                f"{os.environ['AI_FOUNDRY_PROJECT_ENDPOINT']}"
                f"openai/deployments/{os.environ['AI_FOUNDRY_DEPLOYMENT_NAME']}"
            ),
            default_headers={"api-key": os.environ["AI_FOUNDRY_API_KEY"]},
            default_query={"api-version": os.environ["AI_FOUNDRY_API_VERSION"]},
        )

        response = client.chat.completions.create(
            model=os.environ["AI_FOUNDRY_DEPLOYMENT_NAME"],
            messages=[{"role": "user", "content": "Reply with exactly: CONNECTED"}],
            max_tokens=10,
            temperature=0,
        )

        reply = response.choices[0].message.content.strip()
        print(f"  {PASS}  Azure OpenAI responded: '{reply}'")
        print(f"  {PASS}  Model: {os.environ['AI_FOUNDRY_DEPLOYMENT_NAME']}")
        print(f"  {PASS}  API version: {os.environ['AI_FOUNDRY_API_VERSION']}")
        return True

    except Exception as e:
        print(f"  {FAIL}  Connection failed: {e}")
        return False


def test_grader():
    """Test 3: Validate the reward function logic."""
    print(f"\n{SEP}")
    print("TEST 3: Grader / Reward Function")
    print(SEP)

    from agent.grader import compute_reward

    test_cases = [
        # (predicted_cat, predicted_pri, predicted_team, exp_cat, exp_pri, exp_team, expected_score)
        ("Technical", "P1-Critical", "SRE",         "Technical", "P1-Critical", "SRE",         1.0,  "Perfect match"),
        ("Technical", "P2-High",    "SRE",          "Technical", "P1-Critical", "SRE",          0.80, "Priority off by 1, cat+team correct"),
        ("Billing",   "P2-High",    "Billing",      "Technical", "P1-Critical", "SRE",          0.20, "Only priority partial"),
        ("General",   "P4-Low",     "Tier1",        "General",   "P4-Low",      "Tier1",        1.0,  "Perfect low priority"),
        ("Billing",   "P3-Medium",  "Billing",      "Billing",   "P2-High",     "Billing",      0.80, "Priority off by 1 + cat+team ok"),
    ]

    all_ok = True
    for pred_cat, pred_pri, pred_team, exp_cat, exp_pri, exp_team, expected, label in test_cases:
        score = compute_reward(pred_cat, pred_pri, pred_team, exp_cat, exp_pri, exp_team)
        ok = abs(score - expected) < 0.01
        status = PASS if ok else FAIL
        print(f"  {status}  [{label}] score={score:.2f} (expected {expected:.2f})")
        if not ok:
            all_ok = False

    return all_ok


def test_agent_end_to_end():
    """Test 4: Full agent call on one ticket."""
    print(f"\n{SEP}")
    print("TEST 4: Agent End-to-End (One Real Ticket)")
    print(SEP)

    try:
        from openai import OpenAI
        from agent.triage_agent import TRIAGE_TOOL, BASELINE_PROMPT, DEPLOYMENT, get_azure_client

        client = get_azure_client()

        test_ticket = {
            "text": "URGENT: Production database is down. All users are affected. Revenue impact is severe.",
            "customer_tier": "Enterprise",
            "expected_category": "Technical",
            "expected_priority": "P1-Critical",
            "expected_team": "SRE",
        }

        print(f"  Ticket: '{test_ticket['text'][:60]}...'")
        print(f"  Tier  : {test_ticket['customer_tier']}")
        print(f"  Calling Azure OpenAI GPT-4.1...")

        user_prompt = BASELINE_PROMPT.format(
            customer_tier=test_ticket["customer_tier"],
            ticket_text=test_ticket["text"],
        )

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
        if not tool_calls:
            print(f"  {FAIL}  No tool call returned by model")
            return False

        args = json.loads(tool_calls[0].function.arguments)
        print(f"\n  Result:")
        print(f"    Category  : {args.get('category', 'N/A')}")
        print(f"    Priority  : {args.get('priority', 'N/A')}")
        print(f"    Team      : {args.get('team', 'N/A')}")
        print(f"    Reasoning : {args.get('reasoning', 'N/A')[:80]}")

        from agent.grader import compute_reward
        reward = compute_reward(
            args.get("category", ""), args.get("priority", ""), args.get("team", ""),
            test_ticket["expected_category"], test_ticket["expected_priority"], test_ticket["expected_team"],
        )
        print(f"\n  {PASS}  Agent returned a valid triage (reward = {reward:.2f}/1.00)")
        return True

    except Exception as e:
        print(f"  {FAIL}  Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agentlightning_import():
    """Test 5: Check agentlightning is installed."""
    print(f"\n{SEP}")
    print("TEST 5: Agent-Lightning Installation")
    print(SEP)
    try:
        import agentlightning as agl
        version = getattr(agl, "__version__", "unknown")
        print(f"  {PASS}  agentlightning installed (version: {version})")
        return True
    except ImportError:
        print(f"  {FAIL}  agentlightning not installed!")
        print(f"         Run: pip install agentlightning>=0.3.0")
        return False


def main():
    print("=" * 55)
    print("  Customer Support Triage Agent — Setup Validator")
    print("=" * 55)

    results = {
        "Environment Variables":       test_env(),
        "Azure OpenAI Connection":     test_azure_connection(),
        "Grader Reward Function":      test_grader(),
        "Agent End-to-End":            test_agent_end_to_end(),
        "Agent-Lightning Installed":   test_agentlightning_import(),
    }

    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)

    passed = 0
    for name, result in results.items():
        status = PASS if result else FAIL
        print(f"  {status}  {name}")
        if result:
            passed += 1

    total = len(results)
    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print(f"\n  🎉 All tests passed! Run: python train_apo.py")
    elif passed >= 4:
        print(f"\n  ⚠️  Almost ready. Fix the failed test above, then run train_apo.py")
    else:
        print(f"\n  🛑 Setup incomplete. Fix the issues above before training.")

    print()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
