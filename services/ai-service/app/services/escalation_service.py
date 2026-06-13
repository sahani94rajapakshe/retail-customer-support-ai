"""Escalation rules — angry customers and high-priority complaints."""

from app.config import ESCALATION_SENTIMENT_THRESHOLD


def check_escalation(
    intent: str,
    sentiment: str,
    sentiment_score: float,
) -> tuple[bool, str | None]:
    if sentiment == "Negative" and sentiment_score >= ESCALATION_SENTIMENT_THRESHOLD:
        return True, f"Negative sentiment score {sentiment_score:.2f} exceeds threshold"

    if intent == "Complaint" and sentiment == "Negative":
        return True, "Complaint with negative sentiment"

    if intent in ("PaymentIssue", "Refund") and sentiment == "Negative":
        return True, f"High-risk intent ({intent}) with negative sentiment"

    return False, None
