"""Tests for the canonical contextual retrieval representation."""

from sova.index_text import contextualized_text


def test_contextualized_text_uses_title_chain_and_context():
    result = contextualized_text(
        "manual",
        "Chapter > Section > Detail",
        "Raw source remains intact.",
        context="A distinguishing retrieval sentence.",
    )

    assert result == (
        "[manual | Chapter > Section > Detail]\n\n"
        "A distinguishing retrieval sentence.\n\n"
        "Raw source remains intact."
    )


def test_contextualized_text_omits_empty_optional_fields():
    assert contextualized_text("notes", None, "Body", context="  ") == (
        "[notes]\n\nBody"
    )
