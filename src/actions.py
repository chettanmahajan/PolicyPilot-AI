"""The closed set of decisions the assistant is allowed to return.

The assignment PDF only names two of these explicitly (REQUEST_PHOTOS in its
example payload, and NEEDS_MORE_INFORMATION as the required fallback). The full
vocabulary is the set of distinct `resolved_action` values in data/tickets.csv,
which is the only place the complete list appears. Kept as a closed enum so the
LLM cannot invent an action the downstream system would not understand.
"""

from enum import StrEnum


class Action(StrEnum):
    # Cancellations
    CANCEL_AND_REFUND = "CANCEL_AND_REFUND"
    CANNOT_CANCEL_AFTER_DISPATCH = "CANNOT_CANCEL_AFTER_DISPATCH"

    # Damaged goods
    APPROVE_REFUND_OR_REPLACEMENT = "APPROVE_REFUND_OR_REPLACEMENT"
    REQUEST_PHOTOS = "REQUEST_PHOTOS"

    # Defective products
    APPROVE_REPLACEMENT = "APPROVE_REPLACEMENT"
    REQUEST_DEFECT_EVIDENCE = "REQUEST_DEFECT_EVIDENCE"

    # Returns / change of mind
    APPROVE_RETURN = "APPROVE_RETURN"
    REJECT_OPENED_ITEM = "REJECT_OPENED_ITEM"
    REJECT_FOOD_RETURN = "REJECT_FOOD_RETURN"

    # Shipping
    WAIT_AND_TRACK = "WAIT_AND_TRACK"
    OPEN_SHIPPING_INVESTIGATION = "OPEN_SHIPPING_INVESTIGATION"
    OFFER_REPLACEMENT_OR_REFUND = "OFFER_REPLACEMENT_OR_REFUND"

    # Wrong item
    REPLACE_CORRECT_ITEM = "REPLACE_CORRECT_ITEM"

    # Cross-cutting
    REJECT_OUTSIDE_WINDOW = "REJECT_OUTSIDE_WINDOW"
    NEEDS_MORE_INFORMATION = "NEEDS_MORE_INFORMATION"
