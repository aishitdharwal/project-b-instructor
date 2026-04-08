"""
Mock tool implementations for Project B — Session 3.

In a real system these would be API calls to actual databases.
Here we use the mock_data/ JSON files so the pipeline is testable
without external dependencies.

These become the agent's "tool functions" in Week 3 (LangGraph).

Tools:
  lookup_order(order_id)     → order status, product, delivery, eligibility
  lookup_account(customer_id) → tier, spend, account age
  lookup_orders_for_customer(customer_id) → all orders for a customer
"""
import os
import json
from datetime import date, datetime

MOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "mock_data")


def _load_json(filename):
    with open(os.path.join(MOCK_DATA_DIR, filename)) as f:
        return json.load(f)


def lookup_order(order_id: str) -> dict:
    """
    Look up an order by ID.

    Returns order details including return eligibility.
    If order_id is unknown, simulates a realistic 'not found' response.

    Args:
        order_id: e.g. "ORD-445521"

    Returns:
        dict with order details and return_eligible flag, or error dict
    """
    orders = _load_json("orders.json")
    order = next((o for o in orders if o["order_id"] == order_id), None)

    if not order:
        return {
            "found": False,
            "order_id": order_id,
            "error": f"Order {order_id} not found. Please verify the order ID.",
        }

    # Calculate return eligibility
    order_date = datetime.strptime(order["date"], "%Y-%m-%d").date()
    days_since_order = (date.today() - order_date).days
    return_window = 15 if order.get("promotional") else 30
    return_eligible = days_since_order <= return_window and order["status"] == "delivered"

    return {
        "found": True,
        "order_id": order["order_id"],
        "customer_id": order["customer_id"],
        "product": order["product"],
        "price": order["price"],
        "order_date": order["date"],
        "delivery_date": order.get("delivery_date"),
        "status": order["status"],
        "promotional": order.get("promotional", False),
        "return_window_days": return_window,
        "days_since_order": days_since_order,
        "return_eligible": return_eligible,
        "return_note": (
            f"Eligible — {return_window - days_since_order} days remaining"
            if return_eligible
            else f"Window closed — {days_since_order} days since order (window: {return_window} days)"
        ),
    }


def lookup_account(customer_id: str) -> dict:
    """
    Look up customer account details.

    Args:
        customer_id: e.g. "CUST001"

    Returns:
        dict with tier, spend, account age, or error dict
    """
    customers = _load_json("customers.json")
    customer = next((c for c in customers if c["id"] == customer_id), None)

    if not customer:
        return {
            "found": False,
            "customer_id": customer_id,
            "error": f"Customer {customer_id} not found.",
        }

    since = datetime.strptime(customer["since"], "%Y-%m-%d").date()
    account_age_days = (date.today() - since).days
    account_age_years = round(account_age_days / 365, 1)

    # Tier-specific return windows
    return_windows = {"gold": 60, "silver": 45, "standard": 30}

    return {
        "found": True,
        "customer_id": customer["id"],
        "name": customer["name"],
        "tier": customer["tier"],
        "annual_spend": customer["annual_spend"],
        "member_since": customer["since"],
        "account_age_years": account_age_years,
        "return_window_days": return_windows.get(customer["tier"], 30),
        "tier_note": (
            f"Premium {customer['tier'].capitalize()} — {return_windows.get(customer['tier'], 30)}-day return window"
            if customer["tier"] != "standard"
            else "Standard membership — 30-day return window"
        ),
    }


def lookup_orders_for_customer(customer_id: str) -> list[dict]:
    """
    Get all orders for a customer (for context in complex queries).

    Args:
        customer_id: e.g. "CUST001"

    Returns:
        List of order dicts (may be empty)
    """
    orders = _load_json("orders.json")
    customer_orders = [o for o in orders if o["customer_id"] == customer_id]

    results = []
    for order in customer_orders:
        order_date = datetime.strptime(order["date"], "%Y-%m-%d").date()
        days_since = (date.today() - order_date).days
        results.append({
            "order_id": order["order_id"],
            "product": order["product"],
            "date": order["date"],
            "status": order["status"],
            "days_since_order": days_since,
            "promotional": order.get("promotional", False),
        })

    return results


def format_tool_result(tool_name: str, result: dict | list) -> str:
    """
    Format a tool result as a concise text summary for the LLM context.
    """
    if isinstance(result, list):
        if not result:
            return f"[{tool_name}]: No results found."
        lines = [f"[{tool_name}]: {len(result)} records found"]
        for r in result:
            lines.append(f"  - {r.get('order_id', '?')}: {r.get('product', '?')} ({r.get('status', '?')})")
        return "\n".join(lines)

    if not result.get("found", True):
        return f"[{tool_name}]: {result.get('error', 'Not found')}"

    lines = [f"[{tool_name}]:"]
    for k, v in result.items():
        if k not in ("found", "customer_id") and v is not None:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Mock Tools Test\n")

    order = lookup_order("ORD-445521")
    print("Order lookup:")
    print(format_tool_result("order_tracker", order))

    print()
    account = lookup_account("CUST001")
    print("Account lookup:")
    print(format_tool_result("account_lookup", account))

    print()
    orders = lookup_orders_for_customer("CUST001")
    print("Customer orders:")
    print(format_tool_result("customer_orders", orders))
