"""
app/tools/account_lookup.py

Mock fintech tool — simulates account / transaction status lookups.
In a real deployment this would call your CRM / core banking API.
"""

from __future__ import annotations

from typing import Optional

# ── Fake account database ──────────────────────────────────────────────────
_ACCOUNTS: dict[str, dict] = {
    "ACC-1001": {
        "holder": "Alex Johnson",
        "type": "Checking",
        "status": "Active",
        "balance": "$4,250.00",
        "last_transaction": "Card payment to Amazon · $89.99 · 2 hours ago",
        "flags": [],
    },
    "ACC-1002": {
        "holder": "Priya Mehta",
        "type": "Savings",
        "status": "Active",
        "balance": "$22,180.50",
        "last_transaction": "Interest credit · $95.23 · yesterday",
        "flags": [],
    },
    "ACC-1003": {
        "holder": "Sam Torres",
        "type": "Checking",
        "status": "Temporarily Frozen",
        "balance": "$1,002.40",
        "last_transaction": "Unrecognized charge · $340.00 · 6 hours ago",
        "flags": ["FRAUD_REVIEW"],
    },
}

# ── Fake transaction dispute log ───────────────────────────────────────────
_DISPUTES: dict[str, dict] = {
    "DIS-5001": {
        "account": "ACC-1001",
        "amount": "$89.99",
        "status": "Under Investigation",
        "opened": "2026-06-30",
        "eta": "July 10, 2026",
    },
    "DIS-5002": {
        "account": "ACC-1003",
        "amount": "$340.00",
        "status": "Provisional Credit Issued",
        "opened": "2026-07-01",
        "eta": "July 11, 2026",
    },
}


def get_account_status(account_id: str) -> str:
    """Return a human-readable account summary for the voice agent to read out."""
    rec = _ACCOUNTS.get(account_id.upper())
    if not rec:
        return f"I couldn't find an account with the ID {account_id}. Please double-check the number and try again."

    flags_note = ""
    if "FRAUD_REVIEW" in rec["flags"]:
        flags_note = " Your account has been temporarily frozen pending a fraud review. Our team will contact you within 24 hours."

    return (
        f"Account {account_id} belongs to {rec['holder']}. "
        f"It's a {rec['type']} account and is currently {rec['status']}. "
        f"Available balance: {rec['balance']}. "
        f"Most recent transaction: {rec['last_transaction']}."
        f"{flags_note}"
    )


def get_dispute_status(dispute_id: str) -> str:
    """Return a human-readable dispute status."""
    rec = _DISPUTES.get(dispute_id.upper())
    if not rec:
        return f"I couldn't locate a dispute with reference {dispute_id}. Please check the reference number from your confirmation email."

    return (
        f"Dispute {dispute_id} for {rec['amount']} was opened on {rec['opened']}. "
        f"Current status: {rec['status']}. "
        f"Expected resolution by {rec['eta']}."
    )
