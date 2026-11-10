"""Build the committed test set: data/testset.jsonl (SYNTHETIC inputs, seeded).

Each row: {"id", "schema", "text", "gold"}. Texts are generated from varied
templates with random fake values, so gold labels are exact. Run once; the
output file is committed so every eval run scores the same inputs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

R = random.Random(20261107)
FIRST = [
    "Maya",
    "Jonas",
    "Priya",
    "Diego",
    "Aiko",
    "Tomas",
    "Fatima",
    "Liam",
    "Chen",
    "Olu",
    "Sofia",
    "Ravi",
]
LAST = [
    "Okafor",
    "Lindqvist",
    "Raman",
    "Herrera",
    "Tanaka",
    "Novak",
    "Haddad",
    "Byrne",
    "Wei",
    "Adeyemi",
]
COMP = [
    "Northwind Labs",
    "Bluefin Logistics",
    "Acme Robotics",
    "Kestrel Health",
    "Orbit Foods",
    "Lumen Legal",
]
DOMAINS = ["example.com", "mail.test", "corp.example"]
PRODUCTS = [
    "Aero X2 headphones",
    "TrailLite tent",
    "Brewmate kettle",
    "PixelPad tablet",
    "Nimbus desk lamp",
]
PROS = [
    "long battery life",
    "lightweight",
    "easy setup",
    "great sound",
    "sturdy build",
    "good price",
]
CONS = ["short cable", "loud fan", "pricey", "flimsy case", "slow charging", "weak app"]
ITEMS = [
    "consulting hours",
    "server rack",
    "USB-C cables",
    "design retainer",
    "license seats",
    "shipping",
]


def name():
    return f"{R.choice(FIRST)} {R.choice(LAST)}"


def contact(i):
    n, c = name(), R.choice(COMP)
    email = n.lower().replace(" ", ".") + "@" + R.choice(DOMAINS)
    phone = f"+1-{R.randint(200, 989)}-{R.randint(200, 989)}-{R.randint(1000, 9999)}"
    t = R.choice(
        [
            f"Hi, I'm {n} from {c}. Reach me at {email} or call {phone}.",
            f"Contact: {n} ({c}). Phone {phone}. Email: {email}",
            f"Please loop in {n} - they handle vendor stuff at {c}; {email}, tel {phone}.",
            f"Signature block:\n{n}\n{c}\n{phone}\n{email}",
        ]
    )
    return t, {"name": n, "email": email, "phone": phone, "company": c}


def invoice(i):
    cur, sym = R.choice([("USD", "$"), ("EUR", "EUR "), ("GBP", "GBP ")])
    k = R.randint(1, 3)
    items, lines = [], []
    for _ in range(k):
        d, q, p = R.choice(ITEMS), R.randint(1, 12), round(R.uniform(5, 400), 2)
        items.append({"description": d, "quantity": q, "unit_price": p})
        lines.append(f"- {d}: {q} x {sym}{p:.2f}")
    total = round(sum(x["quantity"] * x["unit_price"] for x in items), 2)
    num, v = f"INV-{R.randint(1000, 9999)}", R.choice(COMP)
    t = (
        f"Invoice {num} from {v}\n"
        + "\n".join(lines)
        + f"\nTotal due: {sym}{total:.2f} ({cur})"
    )
    return t, {
        "invoice_number": num,
        "vendor": v,
        "currency": cur,
        "total": total,
        "line_items": items,
    }


def event(i):
    people = [name() for _ in range(R.randint(1, 4))]
    title = R.choice(
        [
            "Quarterly planning",
            "Design review",
            "Board sync",
            "Launch retro",
            "Hiring panel",
        ]
    )
    date = f"2026-{R.randint(1, 12):02d}-{R.randint(1, 28):02d}"
    virt = R.random() < 0.5
    where = "on Zoom" if virt else "in the Chicago office, room 4B"
    t = R.choice(
        [
            f"{title} is set for {date} {where}. Invited: {', '.join(people)}.",
            f"Reminder: {', '.join(people)} - {title.lower()} on {date}, {where}.",
        ]
    )
    return t, {"title": title, "date": date, "attendees": people, "is_virtual": virt}


def ticket(i):
    cid = f"CUST-{R.randint(10000, 99999)}"
    cat, pri, s = R.choice(
        [
            (
                "billing",
                "high",
                "I was charged twice for my March subscription and need a refund.",
            ),
            (
                "technical",
                "urgent",
                "The API returns 500 errors on every request since this morning; production is down.",
            ),
            (
                "account",
                "medium",
                "I can't change the email address on my account, the save button does nothing.",
            ),
            (
                "technical",
                "low",
                "Dark mode text is slightly hard to read on the settings page.",
            ),
            ("billing", "medium", "Can you send a VAT invoice for my last payment?"),
            ("other", "low", "Do you have any plans to open an office in Lisbon?"),
            (
                "account",
                "urgent",
                "Someone logged into my account from another country; please lock it now.",
            ),
        ]
    )
    t = R.choice(
        [
            f"From customer {cid}: {s}",
            f"Ticket opened by {cid}.\nMessage: {s}",
        ]
    )
    return t, {"customer_id": cid, "category": cat, "priority": pri, "summary": s[:120]}


def review(i):
    p = R.choice(PRODUCTS)
    rating = R.randint(1, 5)
    pros, cons = R.sample(PROS, R.randint(1, 2)), R.sample(CONS, R.randint(0, 2))
    rec = rating >= 4
    t = (
        f"Review of the {p}: {rating}/5. What I liked: {', '.join(pros)}. "
        + (f"Downsides: {', '.join(cons)}. " if cons else "No real downsides. ")
        + ("I'd recommend it." if rec else "I would not recommend it.")
    )
    return t, {
        "product": p,
        "rating": rating,
        "pros": pros,
        "cons": cons,
        "would_recommend": rec,
    }


GEN = {
    "ContactCard": contact,
    "Invoice": invoice,
    "Event": event,
    "SupportTicket": ticket,
    "ProductReview": review,
}


def main(per_schema: int = 12):
    out = Path(__file__).resolve().parents[1] / "data" / "testset.jsonl"
    rows = []
    for s, fn in GEN.items():
        for i in range(per_schema):
            text, gold = fn(i)
            rows.append({"id": f"{s}-{i:02d}", "schema": s, "text": text, "gold": gold})
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
