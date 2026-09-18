"""Run the decision pipeline against labelled cases and report accuracy.

    python evaluate.py                      # the supplied sample_test_cases.json
    python evaluate.py --csv                # the 214 historical tickets (generalisation)
    python evaluate.py --csv --limit 50     # a cheaper slice of them

Requires GEMINI_API_KEY, since it exercises the real model.

The historical CSV is used only to *score* the pipeline. Nothing in the running
system reads it - DATA_NOTES.md is explicit that decisions must come from the
policies, not from looking up a similar past ticket.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from src.config import PROJECT_ROOT
from src.decision import DecisionUnavailableError, generate_decision
from src.schemas import TicketCreate

SAMPLE_CASES = PROJECT_ROOT / "sample_test_cases.json"
TICKETS_CSV = PROJECT_ROOT / "data" / "tickets.csv"


@dataclass
class Case:
    case_id: str
    ticket: TicketCreate
    expected: str


@dataclass
class Result:
    case: Case
    predicted: str
    confidence: float | None
    reason: str
    sources: list[str]

    @property
    def correct(self) -> bool:
        return self.predicted == self.case.expected


def _optional_int(value: object) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(float(value))  # type: ignore[arg-type]


def _optional_float(value: object) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)  # type: ignore[arg-type]


def load_sample_cases(path: Path = SAMPLE_CASES) -> list[Case]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        Case(
            case_id=row["case_id"],
            expected=row["expected_action"],
            ticket=TicketCreate(
                message=row["message"],
                order_value_inr=_optional_float(row.get("order_value_inr")),
                days_since_delivery=_optional_int(row.get("days_since_delivery")),
                days_since_dispatch=_optional_int(row.get("days_since_dispatch")),
                product_type=row.get("product_type"),
                opened_status=row.get("opened_status"),
                order_status=row.get("order_status"),
            ),
        )
        for row in rows
    ]


def load_csv_cases(path: Path = TICKETS_CSV, limit: int | None = None) -> list[Case]:
    cases: list[Case] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cases.append(
                Case(
                    case_id=f"T{row['ticket_id']}",
                    expected=row["resolved_action"],
                    ticket=TicketCreate(
                        message=row["message"],
                        order_value_inr=_optional_float(row["order_value_inr"]),
                        days_since_delivery=_optional_int(row["days_since_delivery"]),
                        days_since_dispatch=_optional_int(row["days_since_dispatch"]),
                        product_type=row["product_type"] or None,
                        opened_status=row["opened_status"] or None,
                        order_status=row["order_status"] or None,
                    ),
                )
            )
            if limit and len(cases) >= limit:
                break
    return cases


def run_case(case: Case) -> Result:
    try:
        decision = generate_decision(case.ticket)
    except DecisionUnavailableError as exc:
        return Result(case, predicted="ERROR", confidence=None, reason=str(exc), sources=[])
    return Result(
        case,
        predicted=decision.action.value,
        confidence=decision.confidence,
        reason=decision.reason,
        sources=decision.sources,
    )


def report(results: list[Result]) -> int:
    correct = [r for r in results if r.correct]
    incorrect = [r for r in results if not r.correct]

    print("\n" + "=" * 78)
    print(f"{'CASE':<8} {'EXPECTED':<30} {'PREDICTED':<30} {'':<4}")
    print("-" * 78)
    for r in results:
        mark = "OK " if r.correct else "FAIL"
        print(f"{r.case.case_id:<8} {r.case.expected:<30} {r.predicted:<30} {mark}")

    if incorrect:
        print("\n" + "=" * 78)
        print("FAILURES")
        print("=" * 78)
        for r in incorrect:
            print(f"\n[{r.case.case_id}] expected {r.case.expected}, got {r.predicted}")
            print(f"  ticket    : {r.case.ticket.message}")
            facts = {
                "value": r.case.ticket.order_value_inr,
                "days_delivered": r.case.ticket.days_since_delivery,
                "days_dispatched": r.case.ticket.days_since_dispatch,
                "product": r.case.ticket.product_type,
                "opened": r.case.ticket.opened_status,
                "status": r.case.ticket.order_status,
            }
            print(f"  facts     : {facts}")
            print(f"  confidence: {r.confidence}")
            print(f"  reason    : {r.reason}")
            print(f"  sources   : {r.sources}")

    total = len(results)
    accuracy = (len(correct) / total * 100) if total else 0.0

    print("\n" + "=" * 78)
    print(f"{total} test cases")
    print(f"Correct: {len(correct)}")
    print(f"Incorrect: {len(incorrect)}")
    print(f"Accuracy: {accuracy:.0f}%")
    print("=" * 78)

    return 0 if not incorrect else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="evaluate data/tickets.csv instead")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    parser.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    args = parser.parse_args()

    cases = load_csv_cases(limit=args.limit) if args.csv else load_sample_cases()
    if args.limit and not args.csv:
        cases = cases[: args.limit]

    label = "historical tickets" if args.csv else "sample test cases"
    print(f"Evaluating {len(cases)} {label}...")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(run_case, cases))

    return report(results)


if __name__ == "__main__":
    sys.exit(main())
