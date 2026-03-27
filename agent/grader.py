"""
grader.py
---------
Reward function for the Customer Support Triage Agent.

Scoring strategy (out of 1.0):
  - Category correct:  +0.30
  - Priority correct:  +0.40
  - Team correct:      +0.30

The priority score uses partial credit:
  - Exact match:       +0.40
  - Off by one level:  +0.20 (e.g., predicted P2 when correct is P1)
  - Off by two+:       +0.00
"""

from __future__ import annotations

# Priority ordering (lower index = higher urgency)
PRIORITY_ORDER = ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"]

def _priority_score(predicted: str, expected: str) -> float:
    """Partial credit for priority — penalizes being far off."""
    try:
        pred_idx = PRIORITY_ORDER.index(predicted)
        exp_idx = PRIORITY_ORDER.index(expected)
    except ValueError:
        return 0.0

    diff = abs(pred_idx - exp_idx)
    if diff == 0:
        return 0.40
    elif diff == 1:
        return 0.20
    else:
        return 0.0


def compute_reward(
    predicted_category: str,
    predicted_priority: str,
    predicted_team: str,
    expected_category: str,
    expected_priority: str,
    expected_team: str,
) -> float:
    """
    Compute a [0.0, 1.0] reward for one triage prediction.

    Returns:
        float: Combined score across category (30%), priority (40%), team (30%)
    """
    category_score = 0.30 if predicted_category.strip() == expected_category.strip() else 0.0
    priority_score = _priority_score(predicted_priority.strip(), expected_priority.strip())
    team_score = 0.30 if predicted_team.strip() == expected_team.strip() else 0.0

    total = category_score + priority_score + team_score
    return round(total, 4)

