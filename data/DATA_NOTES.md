# Data Notes

The supplied data is fully synthetic and contains no real customer information.

`tickets.csv` represents historical support tickets. `resolved_action` is the historical decision for that ticket.

Your AI decision endpoint should not simply look up or copy a historical row. It should use the current ticket information and the supplied policy knowledge base to determine an action.

The visible sample test cases are intended only to verify that your implementation works end-to-end.
