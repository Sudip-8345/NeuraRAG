"""
Evaluation question set — 8 questions with expected behavior.

Categories:
  - ANSWERABLE: Answer exists clearly in the policy docs.
  - PARTIALLY_ANSWERABLE: Only part of the answer is in the docs.
  - UNANSWERABLE: No relevant info in the docs at all.
"""

EVAL_QUESTIONS = [
    # --- Answerable (clear answers in docs) ---
    {
        "id": 1,
        "question": "How can I cancel my monthly subscription?",
        "category": "ANSWERABLE",
        "expected_keywords": ["anytime", "account", "support", "end of current billing cycle"],
        "expected_source": "cancellation_policy.md",
        "notes": "Cancellation policy clearly states monthly plans can be cancelled anytime.",
    },
    {
        "id": 2,
        "question": "How long does it take to process a refund?",
        "category": "ANSWERABLE",
        "expected_keywords": ["7-10 business days", "7–10", "original payment method"],
        "expected_source": "refund_policy.md",
        "notes": "Refund policy states 7–10 business days.",
    },
    {
        "id": 3,
        "question": "What happens if I cancel a training workshop within 24 hours of the session?",
        "category": "ANSWERABLE",
        "expected_keywords": ["non-refundable", "no-show"],
        "expected_source": "cancellation_policy.md",
        "notes": "Cancellation within 24 hours or no-show is non-refundable.",
    },
    {
        "id": 4,
        "question": "How are project deliverables shared with clients?",
        "category": "ANSWERABLE",
        "expected_keywords": ["secure repositories", "Git", "cloud repos", "shared drives"],
        "expected_source": "shipping_policy.md",
        "notes": "Shipping/delivery policy describes Git, cloud repos, shared drives.",
    },

    # --- Partially Answerable ---
    {
        "id": 5,
        "question": "What is the refund policy for annual subscriptions and what discounts are available?",
        "category": "PARTIALLY_ANSWERABLE",
        "expected_keywords": ["non-refundable", "not available", "no information"],
        "expected_source": "refund_policy.md",
        "notes": "Refund info exists but discount info does not. Should answer refund part and flag discount part.",
    },
    {
        "id": 6,
        "question": "Can I get a refund if I cancel after the project starts, and who is my account manager?",
        "category": "PARTIALLY_ANSWERABLE",
        "expected_keywords": ["work completed", "not available", "no information", "account manager"],
        "expected_source": "cancellation_policy.md",
        "notes": "Cancellation after start is covered, but account manager assignment is not.",
    },

    # --- Unanswerable (not in docs) ---
    {
        "id": 7,
        "question": "What are the pricing tiers for Neura Dynamics AI platform?",
        "category": "UNANSWERABLE",
        "expected_keywords": ["not available", "not found", "no information", "don't have"],
        "expected_source": None,
        "notes": "Pricing is never mentioned in any policy document.",
    },
    {
        "id": 8,
        "question": "Does Neura Dynamics offer a free trial period?",
        "category": "UNANSWERABLE",
        "expected_keywords": ["not available", "not found", "no information", "don't have"],
        "expected_source": None,
        "notes": "Free trials are never mentioned in any policy document.",
    },
]
