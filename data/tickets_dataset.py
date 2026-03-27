"""
tickets_dataset.py
------------------
Realistic customer support ticket dataset for training and validation.
Each ticket has:
  - text: The raw ticket content from a customer
  - expected_category: Billing | Technical | Account | General
  - expected_priority: P1-Critical | P2-High | P3-Medium | P4-Low
  - expected_team: Tier1 | Tier2 | Billing | SRE | Account-Mgmt
"""

from typing import TypedDict


class TicketTask(TypedDict):
    ticket_id: str
    text: str
    customer_tier: str           # Free | Pro | Enterprise
    expected_category: str
    expected_priority: str
    expected_team: str


# ─── TRAINING DATASET ─────────────────────────────────────────────────────────

TRAIN_DATASET: list[TicketTask] = [
    # ── P1-Critical / SRE ────────────────────────────────────────────────────
    {
        "ticket_id": "T001",
        "text": "URGENT: Our entire production environment is down. All API calls are returning 503. We process $50K/hour in transactions. This is a complete outage affecting all customers.",
        "customer_tier": "Enterprise",
        "expected_category": "Technical",
        "expected_priority": "P1-Critical",
        "expected_team": "SRE",
    },
    {
        "ticket_id": "T002",
        "text": "System completely unresponsive since 3 AM. Database connection pool exhausted. Revenue impact: ~$200K so far. Need immediate escalation.",
        "customer_tier": "Enterprise",
        "expected_category": "Technical",
        "expected_priority": "P1-Critical",
        "expected_team": "SRE",
    },
    # ── P2-High / Tier2 ──────────────────────────────────────────────────────
    {
        "ticket_id": "T003",
        "text": "Our dashboard is loading extremely slowly — takes 45+ seconds. This is affecting our team's ability to monitor campaigns. We're a Pro customer.",
        "customer_tier": "Pro",
        "expected_category": "Technical",
        "expected_priority": "P2-High",
        "expected_team": "Tier2",
    },
    {
        "ticket_id": "T004",
        "text": "The webhook integration stopped sending events 2 hours ago. Our downstream pipeline is now missing data. This is business-critical.",
        "customer_tier": "Pro",
        "expected_category": "Technical",
        "expected_priority": "P2-High",
        "expected_team": "Tier2",
    },
    {
        "ticket_id": "T005",
        "text": "Export functionality broken — every CSV download fails with a 500 error. We need exports for a board meeting tomorrow morning.",
        "customer_tier": "Pro",
        "expected_category": "Technical",
        "expected_priority": "P2-High",
        "expected_team": "Tier2",
    },
    # ── P2-High / Billing ────────────────────────────────────────────────────
    {
        "ticket_id": "T006",
        "text": "We were charged $4,500 this month but our contract cap is $3,000. Please investigate immediately and issue a refund for the difference.",
        "customer_tier": "Enterprise",
        "expected_category": "Billing",
        "expected_priority": "P2-High",
        "expected_team": "Billing",
    },
    {
        "ticket_id": "T007",
        "text": "Double-charged for November subscription. I see two identical charges of $299 on my credit card statement from your company.",
        "customer_tier": "Pro",
        "expected_category": "Billing",
        "expected_priority": "P2-High",
        "expected_team": "Billing",
    },
    # ── P3-Medium / Tier1 ────────────────────────────────────────────────────
    {
        "ticket_id": "T008",
        "text": "I can't figure out how to set up multi-factor authentication for my team. The documentation isn't very clear about the SAML configuration steps.",
        "customer_tier": "Free",
        "expected_category": "Technical",
        "expected_priority": "P3-Medium",
        "expected_team": "Tier1",
    },
    {
        "ticket_id": "T009",
        "text": "How do I export my data as JSON instead of CSV? I looked in the settings but couldn't find the option.",
        "customer_tier": "Free",
        "expected_category": "General",
        "expected_priority": "P3-Medium",
        "expected_team": "Tier1",
    },
    {
        "ticket_id": "T010",
        "text": "The dark mode toggle doesn't seem to be saving my preference. Every time I refresh the page, it goes back to light mode.",
        "customer_tier": "Free",
        "expected_category": "Technical",
        "expected_priority": "P3-Medium",
        "expected_team": "Tier1",
    },
    # ── P3-Medium / Billing ──────────────────────────────────────────────────
    {
        "ticket_id": "T011",
        "text": "Can you please send me an invoice for my last 3 months of payments? I need them for my company's accounting department.",
        "customer_tier": "Pro",
        "expected_category": "Billing",
        "expected_priority": "P3-Medium",
        "expected_team": "Billing",
    },
    {
        "ticket_id": "T012",
        "text": "I'd like to downgrade from Pro to Free plan at the end of this billing cycle. Please confirm what will happen to my data.",
        "customer_tier": "Pro",
        "expected_category": "Billing",
        "expected_priority": "P3-Medium",
        "expected_team": "Billing",
    },
    # ── P3-Medium / Account-Mgmt ─────────────────────────────────────────────
    {
        "ticket_id": "T013",
        "text": "We need to transfer ownership of our Enterprise account to a new admin. Our original admin has left the company.",
        "customer_tier": "Enterprise",
        "expected_category": "Account",
        "expected_priority": "P3-Medium",
        "expected_team": "Account-Mgmt",
    },
    {
        "ticket_id": "T014",
        "text": "Please add 5 new seats to our Enterprise plan and update the billing contact to finance@company.com.",
        "customer_tier": "Enterprise",
        "expected_category": "Account",
        "expected_priority": "P3-Medium",
        "expected_team": "Account-Mgmt",
    },
    # ── P4-Low / Tier1 ───────────────────────────────────────────────────────
    {
        "ticket_id": "T015",
        "text": "Just a general question — do you have a mobile app? I searched the App Store but couldn't find anything.",
        "customer_tier": "Free",
        "expected_category": "General",
        "expected_priority": "P4-Low",
        "expected_team": "Tier1",
    },
    {
        "ticket_id": "T016",
        "text": "Would love a keyboard shortcut to quickly toggle between views. Just a feature suggestion for your team!",
        "customer_tier": "Free",
        "expected_category": "General",
        "expected_priority": "P4-Low",
        "expected_team": "Tier1",
    },
    {
        "ticket_id": "T017",
        "text": "Is there a way to customize the color scheme beyond the preset themes? Would love to match our brand colors.",
        "customer_tier": "Free",
        "expected_category": "General",
        "expected_priority": "P4-Low",
        "expected_team": "Tier1",
    },
    {
        "ticket_id": "T018",
        "text": "Are there any upcoming webinars or training sessions I can join to learn more about advanced features?",
        "customer_tier": "Free",
        "expected_category": "General",
        "expected_priority": "P4-Low",
        "expected_team": "Tier1",
    },
    # ── P1-Critical / SRE ────────────────────────────────────────────────────
    {
        "ticket_id": "T019",
        "text": "Data loss emergency: our analytics pipeline deleted 3 months of production data due to what appears to be a bug in your batch processing API.",
        "customer_tier": "Enterprise",
        "expected_category": "Technical",
        "expected_priority": "P1-Critical",
        "expected_team": "SRE",
    },
    {
        "ticket_id": "T020",
        "text": "Security breach suspected: we're seeing login attempts from unknown IPs accessing our Enterprise account. Possible unauthorized data access.",
        "customer_tier": "Enterprise",
        "expected_category": "Technical",
        "expected_priority": "P1-Critical",
        "expected_team": "SRE",
    },
]


# ─── VALIDATION DATASET ────────────────────────────────────────────────────────

VAL_DATASET: list[TicketTask] = [
    {
        "ticket_id": "V001",
        "text": "Production API returning 429 rate limit errors for the past 30 minutes. All enterprise customers affected. Revenue loss ongoing.",
        "customer_tier": "Enterprise",
        "expected_category": "Technical",
        "expected_priority": "P1-Critical",
        "expected_team": "SRE",
    },
    {
        "ticket_id": "V002",
        "text": "We've been invoiced $12,000 but our agreed annual contract is $10,000. Please correct and issue a credit note.",
        "customer_tier": "Enterprise",
        "expected_category": "Billing",
        "expected_priority": "P2-High",
        "expected_team": "Billing",
    },
    {
        "ticket_id": "V003",
        "text": "The bulk import tool crashes after uploading files larger than 100MB. We have 500MB datasets to process regularly.",
        "customer_tier": "Pro",
        "expected_category": "Technical",
        "expected_priority": "P2-High",
        "expected_team": "Tier2",
    },
    {
        "ticket_id": "V004",
        "text": "Can you explain the difference between the Pro and Enterprise plans? Specifically interested in the API rate limits.",
        "customer_tier": "Free",
        "expected_category": "General",
        "expected_priority": "P4-Low",
        "expected_team": "Tier1",
    },
    {
        "ticket_id": "V005",
        "text": "Need to merge two Enterprise accounts following our company acquisition. What is the process?",
        "customer_tier": "Enterprise",
        "expected_category": "Account",
        "expected_priority": "P3-Medium",
        "expected_team": "Account-Mgmt",
    },
    {
        "ticket_id": "V006",
        "text": "Login button non-functional on Safari 17. Other browsers work fine. Affects our remote team who use Macs exclusively.",
        "customer_tier": "Pro",
        "expected_category": "Technical",
        "expected_priority": "P2-High",
        "expected_team": "Tier2",
    },
    {
        "ticket_id": "V007",
        "text": "Could you add support for Slack notifications when a report is ready? Would be very helpful for my workflow.",
        "customer_tier": "Free",
        "expected_category": "General",
        "expected_priority": "P4-Low",
        "expected_team": "Tier1",
    },
    {
        "ticket_id": "V008",
        "text": "Please cancel my subscription effective immediately and confirm no further charges will be made.",
        "customer_tier": "Pro",
        "expected_category": "Billing",
        "expected_priority": "P3-Medium",
        "expected_team": "Billing",
    },
]


def get_train_dataset() -> list[TicketTask]:
    return TRAIN_DATASET


def get_val_dataset() -> list[TicketTask]:
    return VAL_DATASET
