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


# What situation each action is for. Without this the model sees 15 bare names
# and has to guess from wording - and two are near-synonyms: it once answered an
# approved *damage* claim with OFFER_REPLACEMENT_OR_REFUND, which is the
# *shipping-delay* remedy. The pairing is taken from data/tickets.csv, where
# every action occurs with exactly one issue_type (apart from the two
# cross-cutting ones). Deliberately no thresholds or time windows here: those
# must still come from the retrieved policy text.
ACTION_GUIDE: dict[Action, str] = {
    Action.CANCEL_AND_REFUND: "cancellation request for an order not yet dispatched",
    Action.CANNOT_CANCEL_AFTER_DISPATCH: "cancellation request for an order already dispatched",
    Action.APPROVE_REFUND_OR_REPLACEMENT: "damaged-goods claim that is approved",
    Action.REQUEST_PHOTOS: "damaged-goods claim that needs photographs before approval",
    Action.APPROVE_REPLACEMENT: "defective-product claim that is approved",
    Action.REQUEST_DEFECT_EVIDENCE: "defective-product claim that needs evidence before approval",
    Action.APPROVE_RETURN: "change-of-mind return that is allowed",
    Action.REJECT_OPENED_ITEM: "change-of-mind return refused because the item was opened",
    Action.REJECT_FOOD_RETURN: "change-of-mind return refused because it is a food product",
    Action.WAIT_AND_TRACK: "undelivered order: customer should keep waiting and tracking",
    Action.OPEN_SHIPPING_INVESTIGATION: "undelivered order that needs a shipping investigation",
    Action.OFFER_REPLACEMENT_OR_REFUND: "undelivered order: the shipment never arrived (NOT for damaged goods)",
    Action.REPLACE_CORRECT_ITEM: "wrong item or wrong flavour received",
    Action.REJECT_OUTSIDE_WINDOW: "any claim reported after its policy's time window",
    Action.NEEDS_MORE_INFORMATION: "facts the policy needs are missing or contradictory",
}

# Actions that give the customer money or goods. After photos or defect
# evidence have been requested, none of these may be issued until usable
# evidence exists (see decision.enforce_evidence_requirement).
GRANTING_ACTIONS: frozenset[Action] = frozenset(
    {
        Action.APPROVE_REFUND_OR_REPLACEMENT,
        Action.APPROVE_REPLACEMENT,
        Action.OFFER_REPLACEMENT_OR_REFUND,
        Action.APPROVE_RETURN,
        Action.CANCEL_AND_REFUND,
        Action.REPLACE_CORRECT_ITEM,
    }
)
