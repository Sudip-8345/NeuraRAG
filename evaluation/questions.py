"""
Categories:
  - ANSWERABLE: Answer exists clearly in the policy docs.
  - PARTIALLY_ANSWERABLE: Only part of the answer is in the docs.
  - UNANSWERABLE: No relevant info in the docs at all.
"""

EVAL_QUESTIONS = [
    # --- Answerable ---
    {
        "id": 1,
        "question": "How long does it take to process a refund?",
        "category": "ANSWERABLE",
        "ground_truth": "Approved refunds are typically processed within 7-10 business days. Refunds are issued to the original payment method where possible; otherwise, a credit note or service credit may be provided.",
        "expected_source": "refund_policy.md",
    },

    # --- Partially Answerable ---
    {
        "id": 2,
        "question": "What is the refund policy for annual subscriptions and what discounts are available?",
        "category": "PARTIALLY_ANSWERABLE",
        "ground_truth": "Monthly and annual subscription fees are generally non-refundable once a billing period has started. No information about discounts is available in the policy documents.",
        "expected_source": "refund_policy.md",
    },
    {
        "id": 3,
        "question": "Can I get a refund if I cancel after the project starts, and who is my account manager?",
        "category": "PARTIALLY_ANSWERABLE",
        "ground_truth": "After project start, you pay for all work completed and committed third-party costs. Prepaid unused phases may be partially refunded or credited per contract. Account manager information is not available in the policy documents.",
        "expected_source": "cancellation_policy.md",
    },

    # --- Unanswerable ---
    {
        "id": 4,
        "question": "Does Neura Dynamics offer a free trial period?",
        "category": "UNANSWERABLE",
        "ground_truth": "This information is not available in the provided policy documents. Free trials are never mentioned.",
        "expected_source": None,
    },
    {
        "id": 5,
        "question": "What programming languages and tech stack does Neura Dynamics use internally?",
        "category": "UNANSWERABLE",
        "ground_truth": "This information is not available in the provided policy documents. Internal tech stack details are not covered.",
        "expected_source": None,
    },
]
