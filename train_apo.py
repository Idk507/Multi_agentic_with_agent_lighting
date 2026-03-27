"""
train_apo.py
------------
Main training script for the Customer Support Triage Agent.

Uses Agent-Lightning's Automatic Prompt Optimization (APO) algorithm
to automatically improve the triage prompt using Azure OpenAI GPT-4.1.

Run:
    python train_apo.py

What happens:
  Round 0: Baseline prompt evaluated → e.g., 0.62 accuracy
  Round 1: APO critiques failed cases → generates improved prompt
  Round 2: Improved prompt re-evaluated → e.g., 0.75 accuracy
  ...
  Final optimized prompt is printed and saved to optimized_prompt.txt
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

import agentlightning as agl

from agent.triage_agent import triage_agent, BASELINE_PROMPT
from data.tickets_dataset import get_train_dataset, get_val_dataset

# ─── Load environment ─────────────────────────────────────────────────────────
load_dotenv()

# ─── Azure AsyncOpenAI client for APO algorithm ───────────────────────────────

def get_azure_async_client() -> AsyncOpenAI:
    """
    AsyncOpenAI client configured for Azure OpenAI.
    APO uses async for efficient multi-call prompt optimization.
    """
    return AsyncOpenAI(
        api_key=os.environ["AI_FOUNDRY_API_KEY"],
        base_url=(
            f"{os.environ['AI_FOUNDRY_PROJECT_ENDPOINT']}"
            f"openai/deployments/{os.environ['AI_FOUNDRY_DEPLOYMENT_NAME']}"
        ),
        default_headers={
            "api-key": os.environ["AI_FOUNDRY_API_KEY"],
        },
        default_query={
            "api-version": os.environ["AI_FOUNDRY_API_VERSION"],
        },
    )


def main():
    print("=" * 65)
    print("  Customer Support Triage Agent — APO Training")
    print("  Powered by Microsoft Agent-Lightning + Azure OpenAI GPT-4.1")
    print("=" * 65)
    print()

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = get_train_dataset()
    val_dataset = get_val_dataset()

    print(f"📦 Training tickets : {len(train_dataset)}")
    print(f"📦 Validation tickets: {len(val_dataset)}")
    print()

    # ── APO Algorithm ─────────────────────────────────────────────────────────
    # APO (Automatic Prompt Optimization):
    #   1. Evaluates current prompt on a batch of tasks
    #   2. Uses an LLM to generate a "textual gradient" (natural language critique)
    #   3. Rewrites the prompt using that critique
    #   4. Repeats for `beam_rounds` iterations

    azure_async_client = get_azure_async_client()

    algo = agl.APO(
        client=azure_async_client,
        # Hyperparameters
        val_batch_size=8,          # Evaluate on 8 tickets per round
        gradient_batch_size=4,     # Use 4 failed tickets for critique
        beam_width=2,              # Keep top-2 prompt candidates
        branch_factor=2,           # Generate 2 alternative prompts per critique
        beam_rounds=3,             # 3 rounds of optimization
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = agl.Trainer(
        algorithm=algo,
        n_runners=4,               # Run 4 agents in parallel
        initial_resources={
            "prompt_template": agl.PromptTemplate(BASELINE_PROMPT)
        },
        # TraceToMessages converts spans into OpenAI chat messages
        # so APO's LLM can understand what went wrong
        adapter=agl.TraceToMessages(),
    )

    print("🚀 Starting APO training...")
    print("   Algorithm : Automatic Prompt Optimization (APO)")
    print("   Runners   : 4 parallel agents")
    print("   Rounds    : 3 beam rounds")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(
        agent=triage_agent,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # ── Retrieve and save the optimized prompt ────────────────────────────────
    optimized_prompt = trainer.get_resource("prompt_template")

    print()
    print("=" * 65)
    print("✅ Training Complete!")
    print("=" * 65)
    print()
    print("📝 Optimized Prompt Template:")
    print("-" * 65)
    print(str(optimized_prompt))
    print("-" * 65)

    with open("optimized_prompt.txt", "w") as f:
        f.write(str(optimized_prompt))

    print()
    print("💾 Optimized prompt saved to: optimized_prompt.txt")
    print("   Use this in run_inference.py for best performance.")


if __name__ == "__main__":
    main()
