# 🎯 Customer Support Ticket Triage Agent
### Built with Microsoft Agent-Lightning + Azure OpenAI (GPT-4.1)

---

## 📌 Real-World Use Case

Automatically **classify**, **prioritize**, and **route** incoming customer support tickets using an AI agent that **self-improves its own prompts** via Agent-Lightning's Automatic Prompt Optimization (APO).

### Problem
- Support teams receive hundreds of tickets daily
- Manual triage is slow, inconsistent, and error-prone
- Wrong routing wastes time and damages customer satisfaction

### Solution
An agent that reads each ticket and:
1. **Classifies** the issue type (Billing / Technical / Account / General)
2. **Assigns priority** (P1-Critical / P2-High / P3-Medium / P4-Low)
3. **Routes** to the correct team (Tier1 / Tier2 / Billing / SRE / Account-Mgmt)
4. **Learns and improves** its own routing prompt automatically via APO

---

## 🏗️ Project Structure

```
support_triage_agent/
├── .env                        # Azure OpenAI credentials
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── data/
│   └── tickets_dataset.py      # Ticket dataset (train + val)
├── agent/
│   ├── __init__.py
│   ├── triage_agent.py         # Core Agent-Lightning LitAgent
│   └── grader.py               # Reward function (accuracy scorer)
├── train_apo.py                # Main APO training script
├── run_inference.py            # Run trained agent on new tickets
└── dashboard.py                # Simple CLI dashboard for results
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install agentlightning openai python-dotenv
```

### 2. Set up .env (already configured)
The `.env` file is pre-configured with your Azure OpenAI credentials.

### 3. Run APO Training
```bash
python train_apo.py
```

### 4. Run inference on new tickets
```bash
python run_inference.py
```

---

## 🧠 How Agent-Lightning Works Here

```
┌─────────────────────────────────────────────────────────┐
│                   Agent-Lightning Loop                   │
│                                                          │
│  ┌──────────┐    Tasks     ┌─────────────┐              │
│  │          │ ──────────►  │             │              │
│  │   APO    │              │  Triage     │              │
│  │Algorithm │  Prompt      │  Agent      │              │
│  │ (Brain)  │ ◄──────────  │  (Worker)   │              │
│  │          │    Spans +   │             │              │
│  └──────────┘    Rewards   └─────────────┘              │
│                                                          │
│  Each round: prompt improves → accuracy increases        │
└─────────────────────────────────────────────────────────┘
```

The APO algorithm:
1. Sends ticket tasks to the agent using the current prompt
2. Collects spans (LLM call traces) + rewards (accuracy scores)
3. Generates a "textual gradient" — LLM critique of the prompt
4. Rewrites the prompt to be more accurate
5. Repeats → prompt gets smarter each round
