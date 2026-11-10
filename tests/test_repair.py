import json

from soe.enforcer import Enforcer
from soe.repair import coerce, extract_json_text, repair, repair_syntax
from soe.schema import UnsupportedSchema, to_node

from eval.schemas import ContactCard, Invoice, ProductReview, SupportTicket


def test_extract_from_fence_and_prose():
    t = 'Sure! Here it is:\n```json\n{"a": 1}\n```\nHope that helps.'
    assert json.loads(extract_json_text(t)) == {"a": 1}


def test_extract_balanced_with_braces_in_strings():
    t = 'x {"a": "}{", "b": {"c": 2}} trailing'
    assert json.loads(extract_json_text(t)) == {"a": "}{", "b": {"c": 2}}


def test_extract_none_when_no_object():
    assert extract_json_text("no json here") is None


def test_syntax_trailing_comma_single_quotes_python_literals():
    s = "{'a': True, 'b': None, 'c': [1, 2,],}"
    assert json.loads(repair_syntax(s)) == {"a": True, "b": None, "c": [1, 2]}


def test_syntax_unquoted_keys_and_unclosed():
    s = '{name: "x", items: [1, 2'
    assert json.loads(repair_syntax(s)) == {"name": "x", "items": [1, 2]}


def test_syntax_comments_removed():
    s = '{"a": 1, // comment\n "b": 2 /* block */}'
    assert json.loads(repair_syntax(s)) == {"a": 1, "b": 2}


def test_coerce_types():
    node = to_node(ProductReview)
    data = {
        "Product": "X",
        "rating": "4",
        "pros": "light",
        "cons": None,
        "would_recommend": "Yes",
        "extra": 1,
    }
    out = coerce(data, node)
    assert out == {
        "product": "X",
        "rating": 4,
        "pros": ["light"],
        "cons": [],
        "would_recommend": True,
    }


def test_coerce_enum_case_and_currency_number():
    node = to_node(Invoice)
    data = {
        "invoice_number": "I-1",
        "vendor": "V",
        "currency": "usd",
        "total": "$1,234.50",
        "line_items": [{"description": "d", "quantity": "2", "unit_price": "10"}],
    }
    out = coerce(data, node)
    assert (
        out["currency"] == "USD"
        and out["total"] == 1234.5
        and out["line_items"][0]["quantity"] == 2
    )


def test_repair_stages():
    good = '{"name":"a","email":"b","phone":"c","company":"d"}'
    assert repair(good, ContactCard).stage == "raw"
    assert repair("Here: " + good, ContactCard).stage == "extract"
    assert repair(good.replace('"d"}', '"d",}'), ContactCard).stage == "syntax"
    r = repair(
        '{"customer_id": 12, "category": "Billing", "priority": "HIGH", "summary": "s"}',
        SupportTicket,
    )
    assert r.ok and r.stage == "coerce" and r.value.customer_id == "12"


def test_repair_fails_with_errors():
    r = repair('{"name": "a"}', ContactCard)
    assert not r.ok and r.stage == "fail" and any("email" in e for e in r.errors)


def test_enforcer_retry_uses_errors():
    calls = []

    def fake(msgs):
        calls.append(msgs)
        if len(calls) == 1:
            return '{"name": "a"}'
        assert "email" in msgs[-1]["content"]  # error feedback reached the model
        return '{"name":"a","email":"b","phone":"c","company":"d"}'

    res = Enforcer(generate_text=fake, max_retries=2).run(ContactCard, "text")
    assert res.ok and res.stage == "retry-1" and res.attempts == 2


def test_enforcer_none_strategy_is_strict():
    res = Enforcer(generate_text=lambda m: "```json\n{}\n```").run(
        ContactCard, "t", strategy="none"
    )
    assert not res.ok


def test_schema_unsupported_optional():
    from typing import Optional

    from pydantic import BaseModel

    class M(BaseModel):
        x: Optional[int]

    try:
        to_node(M)
        assert False
    except UnsupportedSchema:
        pass


def test_schema_node_constraints():
    n = to_node(Invoice)
    li = dict(n.properties)["line_items"]
    assert li.kind == "array" and li.min_items == 1 and li.max_items == 5
    q = dict(li.items.properties)["quantity"]
    assert (q.minimum, q.maximum) == (1, 100)
    assert dict(to_node(SupportTicket).properties)["summary"].max_length == 120
