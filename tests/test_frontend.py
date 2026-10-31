"""Tests for the Streamlit frontend, driving the real page against the real API."""

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from app.main import create_app
from app.schemas import PRIORITIES, TAXONOMY, AccessLevel

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
APP = str(FRONTEND / "streamlit_app.py")


@pytest.fixture
def api_url(deps):
    """Serve the app over real HTTP, because the frontend is a separate process."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(deps), port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def page(api_url, monkeypatch):
    """Return a function that opens the page, optionally signed in."""
    monkeypatch.syspath_prepend(str(FRONTEND))
    monkeypatch.setenv("API_BASE_URL", api_url)
    monkeypatch.delenv("TICKET_API_KEY", raising=False)

    def open_page(key="analyst-key"):
        app = AppTest.from_file(APP, default_timeout=30).run()
        if key:
            app.sidebar.text_input(key="api_key").set_value(key).run()
        assert not app.exception
        return app

    return open_page


def fill_ticket(app, sample="VPN certificate expired"):
    app.sidebar.radio(key="page").set_value("Analyze a ticket").run()
    app.selectbox(key="sample").set_value(sample).run()
    return app


def analyze(app):
    app.button(key="analyze").click().run()
    assert not app.exception
    return app


def text_of(elements):
    return " ".join(element.value for element in elements)


def test_prompts_for_a_key_before_showing_anything(page):
    app = page(key=None)
    assert "Connect with an API key" in text_of(app.info)
    assert not app.button


def test_rejects_a_bad_key(page):
    assert "Invalid or missing API key" in text_of(page(key="wrong").sidebar.error)


def test_reports_an_unreachable_api(page, monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    assert "Cannot reach the API" in text_of(page().sidebar.error)


def test_sample_ticket_gets_a_recommendation_with_evidence(page):
    app = analyze(fill_ticket(page()))
    assert "Recommendation ready" in text_of(app.success)
    assert any("Resolution path 1" == item.value for item in app.subheader)
    assert any("Evidence" in item.label for item in app.expander)
    assert app.metric[0].label == "Confidence"


def test_each_submission_gets_a_new_ticket_id(page):
    app = fill_ticket(page())
    first = app.text_input(key="ticket_id").value
    analyze(app)
    assert app.text_input(key="ticket_id").value != first


def test_reusing_an_id_for_different_text_shows_the_api_error(page):
    app = fill_ticket(page())
    used_id = app.text_input(key="ticket_id").value
    analyze(app)
    app.text_input(key="ticket_id").set_value(used_id)
    app.text_input(key="title").set_value("A different ticket title")
    analyze(app)
    assert "already submitted with different content" in text_of(app.error)
    assert app.text_input(key="ticket_id").value == used_id


def test_validation_errors_are_readable(page):
    app = fill_ticket(page())
    app.text_input(key="title").set_value("x")
    analyze(app)
    assert "title" in text_of(app.error)


def test_ticket_above_clearance_is_refused(page):
    app = fill_ticket(page())
    app.selectbox(key="access_level").set_value("RESTRICTED")
    analyze(app)
    assert "clearance" in text_of(app.error)


def submit_feedback(app):
    run_id = app.session_state["run_id"]
    app.selectbox(key=f"verdict-{run_id}").set_value("USEFUL")
    app.button(key=f"submit-{run_id}").click().run()
    assert not app.exception
    return app


def test_analyst_feedback_is_recorded_and_shown(page):
    app = submit_feedback(analyze(fill_ticket(page())))
    assert "analyst marked recommendation as USEFUL" in text_of(app.caption).replace("\\", "")


def test_key_without_review_permission_sees_the_refusal(page):
    app = submit_feedback(analyze(fill_ticket(page(key="service-key"))))
    assert "Reviewer permission required" in text_of(app.error)


def test_run_lookup_shows_a_finished_run_and_a_missing_one(page):
    app = analyze(fill_ticket(page()))
    run_id = app.session_state["run_id"]

    app.sidebar.radio(key="page").set_value("Look up a run").run()
    app.text_input(key="lookup_run_id").set_value(run_id).run()
    assert "Recommendation ready" in text_of(app.success)

    app.text_input(key="lookup_run_id").set_value("missing").run()
    assert "Run not found" in text_of(app.error)


def test_search_lists_everything_then_filters(page):
    app = page()
    app.sidebar.radio(key="page").set_value("Search tickets").run()
    # 12 of the 13 seed tickets: this key's clearance cannot see the RESTRICTED one.
    assert "12 tickets found" in text_of(app.caption)

    app.text_input(key="search_q").set_value("VPN")
    app.button(key="search_submit").click().run()
    assert not app.exception
    titles = app.dataframe[0].value["Title"]
    assert 0 < len(titles) < 12
    assert all("VPN" in title for title in titles)
    assert f"{len(titles)} tickets found" in text_of(app.caption)


def chat_text(app):
    return " ".join(item.value for message in app.chat_message for item in message.markdown).replace("\\", "")


def test_chat_keeps_a_conversation_and_shows_sources(page):
    app = page()
    app.chat_input(key="chat_prompt").set_value("GlobalProtect VPN error GP-4012 expired client certificate").run()
    assert not app.exception
    assert [message.name for message in app.chat_message] == ["user", "assistant"]
    assert "looks related" in chat_text(app)
    assert any("Verified sources" in item.label for item in app.expander)

    app.chat_input(key="chat_prompt").set_value("That did not work").run()
    assert not app.exception
    assert len(app.chat_message) == 4


def test_chat_example_button_sends_a_message(page):
    app = page()
    app.button(key="chat_example_0").click().run()
    assert not app.exception
    assert app.chat_message[0].name == "user"
    assert "GP-4012" in chat_text(app)


def test_new_chat_clears_the_conversation(page):
    app = page()
    app.chat_input(key="chat_prompt").set_value("VPN certificate expired").run()
    app.sidebar.button(key="new_chat").click().run()
    assert not app.exception
    assert not app.chat_message


def test_message_left_unanswered_by_an_interrupted_reply_gets_answered(page):
    app = page()
    app.session_state["chat_turns"] = [{"role": "user", "content": "GlobalProtect VPN GP-4012 expired certificate"}]
    app.run()
    assert not app.exception
    assert [turn["role"] for turn in app.session_state["chat_turns"]] == ["user", "assistant"]


def test_failed_message_is_not_kept_in_the_conversation(page):
    app = page()
    app.chat_input(key="chat_prompt").set_value("x" * 4001).run()
    assert not app.exception
    assert "content" in text_of(app.error)
    assert app.session_state["chat_turns"] == []


def test_form_choices_match_the_api(page):
    page()
    from streamlit_app import ACCESS_LEVELS, CATEGORIES
    from streamlit_app import PRIORITIES as FRONTEND_PRIORITIES

    assert list(CATEGORIES) == list(TAXONOMY)
    assert FRONTEND_PRIORITIES == PRIORITIES
    assert ACCESS_LEVELS == tuple(level.value for level in AccessLevel)


def test_plain_stops_untrusted_text_from_being_markdown(page):
    page()
    from streamlit_app import plain

    escaped = plain("![x](http://evil.test/a.png) [link](http://evil.test) :red[alert]")
    for marker in ("!", "[", "]", "(", ":"):
        assert marker not in escaped.replace(f"\\{marker}", "")
