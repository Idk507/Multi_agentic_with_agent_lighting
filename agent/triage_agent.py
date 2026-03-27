"""
triage_agent.py
---------------
Customer Support Ticket Triage Agent powered by Agent-Lightning + Azure OpenAI.

This agent:
  1. Receives a support ticket (text + customer tier)
  2. Calls GPT-4.1 via Azure OpenAI to classify/prioritize/route it
  3. Uses function-calling (tool use) to submit the structured triage result
  4. Returns a reward score via the grader

The @agl.rollout decorator makes this function a trainable Agent-Lightning rollout.
The APO algorithm will automatically improve `prompt_template` across training rounds.
"""

from __future__ import annotations

import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

import agentlightning as agl

from .grader import compute_reward

load_dotenv()

# ─── Azure OpenAI client ──────────────────────────────────────────────────────

def get_azure_client() -> OpenAI:
    """Return an OpenAI client configured for Azure OpenAI."""
    return OpenAI(
        api_key=os.environ["AI_FOUNDRY_API_KEY"],
        base_url=f"{os.environ['AI_FOUNDRY_PROJECT_ENDPOINT']}openai/deployments/{os.environ['AI_FOUNDRY_DEPLOYMENT_NAME']}",
        default_headers={
            "api-key": os.environ["AI_FOUNDRY_API_KEY"],
        },
        default_query={
            "api-version": os.environ["AI_FOUNDRY_API_VERSION"],
        },
    )

DEPLOYMENT = os.environ.get("AI_FOUNDRY_DEPLOYMENT_NAME", "gpt-4.1")

# ─── Tool definition ─────────────────────────────────────────────────────────

TRIAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_triage",
        "description": (
            "Submit the triage classification for a customer support ticket. "
            "Call this once you have analyzed the ticket and made your decisions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["Billing", "Technical", "Account", "General"],
                    "description": "The category that best describes the issue.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"],
                    "description": (
                        "Priority level. "
                        "P1-Critical: complete outage / data loss / security breach / revenue > $10K/hr impact. "
                        "P2-High: major degradation / billing overcharge > $500 / Pro/Enterprise workflow blocker. "
                        "P3-Medium: partial issue / standard billing request / account management. "
                        "P4-Low: general questions / feature requests / cosmetic issues."
                    ),
                },
                "team": {
                    "type": "string",
                    "enum": ["SRE", "Tier2", "Billing", "Account-Mgmt", "Tier1"],
                    "description": (
                        "Routing target. "
                        "SRE: production outages, data loss, security breaches. "
                        "Tier2: complex bugs, integrations, Enterprise/Pro technical issues. "
                        "Billing: all payment, invoice, refund, subscription queries. "
                        "Account-Mgmt: Enterprise account changes, seat management, ownership transfers. "
                        "Tier1: general questions, feature requests, basic how-to queries."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief (1-2 sentence) explanation of your triage decision.",
                },
            },
            "required": ["category", "priority", "team", "reasoning"],
        },
    },
}

# ─── Result type ─────────────────────────────────────────────────────────────

class TriageResult(TypedDict):
    category: str
    priority: str
    team: str
    reasoning: str
    reward: float


# ─── Baseline prompt template ────────────────────────────────────────────────

BASELINE_PROMPT = """You are an expert customer support triage specialist. Analyze the incoming support ticket and classify it accurately.

Customer Tier: {customer_tier}

Ticket Content:
{ticket_text}

Analyze the ticket carefully and use the submit_triage tool to record your classification. Consider:
- The urgency and business impact described
- The customer tier (Enterprise > Pro > Free for priority weighting)
- The nature of the problem (technical/billing/account/general)
- Which team is best equipped to handle this"""


# ─── Agent-Lightning rollout ─────────────────────────────────────────────────

@agl.rollout
def triage_agent(task: dict, prompt_template: agl.PromptTemplate) -> float:
    """
    Agent-Lightning rollout for the triage agent.

    Args:
        task: A TicketTask dict from the dataset
        prompt_template: The current (potentially APO-optimized) prompt

    Returns:
        float: Reward score [0.0, 1.0]
    """
    client = get_azure_client()

    # Format the prompt with task-specific values
    user_prompt = str(prompt_template).format(
        customer_tier=task["customer_tier"],
        ticket_text=task["text"],
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise customer support triage agent. "
                "Always use the submit_triage function to record your decision. "
                "Be concise and accurate."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    # ── First LLM call: agent decides action ─────────────────────────────────
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        tools=[TRIAGE_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_triage"}},
        temperature=0.1,   # Low temperature for deterministic classification
        max_tokens=512,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # ── Parse tool call result ────────────────────────────────────────────────
    result: TriageResult = {
        "category": "General",
        "priority": "P4-Low",
        "team": "Tier1",
        "reasoning": "Fallback — no tool call returned.",
        "reward": 0.0,
    }

    if tool_calls:
        tool_call = tool_calls[0]
        try:
            args = json.loads(tool_call.function.arguments)
            result["category"] = args.get("category", "General")
            result["priority"] = args.get("priority", "P4-Low")
            result["team"] = args.get("team", "Tier1")
            result["reasoning"] = args.get("reasoning", "")
        except (json.JSONDecodeError, KeyError):
            pass  # Use fallback values

    # ── Compute reward ────────────────────────────────────────────────────────
    reward = compute_reward(
        predicted_category=result["category"],
        predicted_priority=result["priority"],
        predicted_team=result["team"],
        expected_category=task["expected_category"],
        expected_priority=task["expected_priority"],
        expected_team=task["expected_team"],
    )
    result["reward"] = reward

    # Emit structured result as a span for APO to analyze
    agl.emit_output(result)

    return reward
