from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_route_case():
    module_path = Path.cwd() / "routing_policy.py"
    spec = importlib.util.spec_from_file_location("routing_policy", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load routing_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.route_case


def main() -> int:
    route_case = _load_route_case()
    cases = [
        (
            {
                "incident_code": "SEC-441",
                "labels": ["login", "credential_leak"],
                "severity": "p2",
                "customer_tier": "standard",
                "component": "auth",
                "region": "us",
            },
            "containment",
        ),
        (
            {
                "incident_code": "OPS-920",
                "labels": ["schema"],
                "severity": "p2",
                "customer_tier": "enterprise",
                "component": "migration",
                "region": "us",
            },
            "principal_architect",
        ),
        (
            {
                "incident_code": "BILL-100",
                "labels": ["refund"],
                "amount_usd": 2500,
                "severity": "p3",
                "customer_tier": "growth",
                "component": "billing",
                "region": "us",
            },
            "revenue_guardian",
        ),
        (
            {
                "incident_code": "APP-730",
                "labels": ["android"],
                "severity": "p1",
                "customer_tier": "standard",
                "component": "mobile",
                "symptom": "crash_loop",
                "region": "us",
            },
            "mobile_hotfix",
        ),
        (
            {
                "incident_code": "REQ-812",
                "labels": ["privacy"],
                "data_subject_request": True,
                "severity": "p3",
                "customer_tier": "standard",
                "component": "account",
                "region": "eu",
            },
            "privacy_review",
        ),
        (
            {
                "incident_code": "OPS-101",
                "labels": ["docs"],
                "severity": "p4",
                "customer_tier": "standard",
                "component": "docs",
                "region": "us",
            },
            "standard",
        ),
    ]
    failures = []
    for index, (ticket, expected) in enumerate(cases, start=1):
        actual = route_case(ticket)
        if actual != expected:
            failures.append(f"case {index}: expected {expected!r}, got {actual!r}")
    if failures:
        raise AssertionError("\n".join(failures))
    print(f"{len(cases)} passed in hidden long-context check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
