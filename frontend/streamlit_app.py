"""Streamlit console for IT analysts.

Run with ``streamlit run frontend/streamlit_app.py``, with the API already running.
This is only a client of the API, so authentication, redaction and clearance checks
all stay in the backend. Ticket and LLM text is untrusted, so it goes through
``plain`` before reaching Markdown.
"""

import os
import re
import time
import uuid
from datetime import date, datetime, time as day_time
from typing import Any

import pandas as pd
import streamlit as st
from api_client import ApiClient, ApiError

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
POLL_SECONDS = 1.0
POLL_TIMEOUT_SECONDS = 120.0
PAGE_SIZE = 25
RECOMMENDATION_REF = "recommendation"
CHAT_HISTORY_MESSAGES = 20
CHAT_MESSAGE_CHARS = 4000
CHAT_EXAMPLES = (
    "GlobalProtect VPN shows error GP-4012 and says my client certificate expired",
    "My mailbox is full and I can't receive email",
    "I'm stuck in a sign-in loop on the HR portal",
)

ACCESS_LEVELS = ("PUBLIC", "INTERNAL", "RESTRICTED")
ENVIRONMENTS = ("", "prod", "staging", "dev")
CATEGORIES = (
    "Access Management",
    "Application Failure",
    "Configuration",
    "Infrastructure",
    "Email and Collaboration",
)
PRIORITIES = ("P1", "P2", "P3", "P4")
TICKET_STATUSES = ("OPEN", "RESOLVED", "CLOSED")
VERDICTS = ("USEFUL", "INCORRECT", "UNRESOLVED")
FINISHED_STATUSES = ("COMPLETED", "FAILED")

OUTCOME_BANNERS = {
    "RECOMMEND": (st.success, "Recommendation ready for analyst review"),
    "CLARIFY": (st.info, "More information needed from the reporter"),
    "ESCALATE": (st.warning, "Escalate to an analyst"),
}
TICKET_FIELDS = ("title", "description", "symptoms", "affected_service")
SAMPLES = {
    "VPN certificate expired": {
        "title": "GlobalProtect VPN fails with GP-4012",
        "description": "Laptop cannot reach the VPN. GlobalProtect shows GP-4012, client certificate expired.",
        "symptoms": "GP-4012 on connect",
        "affected_service": "VPN",
        "environment": "prod",
    },
    "Vague report": {
        "title": "It is not working",
        "description": "Something is broken, please help.",
        "symptoms": "",
        "affected_service": "Unknown",
        "environment": "",
    },
    "Nothing similar on file": {
        "title": "Badge reader offline at the loading dock",
        "description": "The door badge reader at the loading dock shows a red light and ignores every card.",
        "symptoms": "Red light on reader",
        "affected_service": "Physical Security",
        "environment": "prod",
    },
}


def plain(text: Any) -> str:
    """Escape Markdown, so untrusted text cannot add links, images or formatting."""
    return re.sub(r"([!-/:-@\[-`{-~])", r"\\\1", str(text))


def new_ticket_id() -> str:
    return f"INC-{uuid.uuid4().hex[:8].upper()}"


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def load_sample() -> None:
    """Copy the chosen sample into the form. Runs as a widget callback."""
    sample = SAMPLES.get(st.session_state.sample)
    if sample is None:
        return
    for field in (*TICKET_FIELDS, "environment"):
        st.session_state[field] = sample[field]
    st.session_state.ticket_id = new_ticket_id()


def connect() -> ApiClient | None:
    """Render the sidebar connection panel. Return a client once the key works."""
    api_key = st.sidebar.text_input(
        "API key",
        value=os.environ.get("TICKET_API_KEY", ""),
        type="password",
        key="api_key",
        help="A key from API_KEYS in .env. Feedback needs a key with can_review.",
    ).strip()
    if not api_key:
        st.sidebar.info("Enter an API key to connect.")
        return None

    client = ApiClient(API_BASE_URL, api_key)
    try:
        client.health()
        client.search(limit=1)
    except ApiError as exc:
        client.close()
        st.sidebar.error(plain(exc))
        return None
    st.sidebar.success(f"Connected to {plain(API_BASE_URL)}")
    return client


def wait_for_run(client: ApiClient, run_id: str) -> dict[str, Any]:
    """Poll until the run finishes or the timeout passes, and return its latest state."""
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    run = client.get_run(run_id)
    if run["status"] in FINISHED_STATUSES:
        return run
    with st.spinner("Analyzing the ticket..."):
        while run["status"] not in FINISHED_STATUSES and time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            run = client.get_run(run_id)
    return run


def show_matches(heading: str, matches: list[dict[str, Any]]) -> None:
    if not matches:
        return
    st.subheader(heading)
    rows = [
        {
            "Source": match["source_id"],
            "Type": match["source_type"],
            "Title": match["title"],
            "Score": match["score"],
            "Status": match.get("status"),
            "Matched on": ", ".join(match["matched_on"]),
        }
        for match in matches
    ]
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"Score": st.column_config.NumberColumn(format="%.2f")},
    )


def show_recommendation(result: dict[str, Any]) -> None:
    for index, path in enumerate(result["recommendation"], start=1):
        st.subheader(f"Resolution path {index}")
        st.markdown(plain(path["summary"]))
        st.markdown("\n".join(f"{n}. {plain(step)}" for n, step in enumerate(path["steps"], start=1)))
        with st.expander(f"Evidence ({len(path['evidence'])})"):
            for claim in path["evidence"]:
                st.markdown(f"**{plain(claim['source_id'])}** ({plain(claim['source_type'].title())})")
                st.markdown("> " + plain(claim["excerpt"]).replace("\n", "\n> "))
                st.caption(plain(claim["reason"]))


def show_details(result: dict[str, Any]) -> None:
    """Show how the outcome was reached."""
    with st.expander("How this was decided"):
        classification = result["classification"]
        if classification:
            st.markdown(f"**Intent:** {plain(classification['intent'])}")
            st.caption(plain(classification["rationale"]))
        if result["confidence_breakdown"]:
            st.markdown("**Confidence signals** (the lowest one sets the overall score)")
            columns = st.columns(len(result["confidence_breakdown"]))
            for column, (name, value) in zip(columns, result["confidence_breakdown"].items()):
                column.metric(name.replace("_", " ").capitalize(), percent(value))
        citations = result["citations"]
        st.markdown(f"**Citations verified:** {citations['claims_verified']} of {citations['claims_total']}")
        for rejected in citations["rejected"]:
            st.caption("Rejected: " + plain(", ".join(f"{key}: {value}" for key, value in rejected.items())))


def feedback_targets(result: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Map a display label to the (target_type, target_ref) the API expects."""
    targets = {}
    if result["recommendation"]:
        targets["The recommendation"] = ("RECOMMENDATION", RECOMMENDATION_REF)
    for match in result["duplicates"] + result["similar_cases"]:
        targets[f"Match: {match['source_id']}"] = ("MATCH", match["source_id"])
    return targets


def show_feedback(client: ApiClient, run: dict[str, Any], targets: dict[str, tuple[str, str]]) -> None:
    st.subheader("Analyst feedback")
    for item in run["feedback"]:
        st.caption(
            f"{plain(item['reviewer_id'])} marked {plain(item['target_ref'])} as {plain(item['verdict'])}"
            + (f": {plain(item['comment'])}" if item["comment"] else "")
        )
    if not targets:
        st.caption("This run has nothing to rate.")
        return

    with st.form(f"feedback-{run['run_id']}", clear_on_submit=True):
        target = st.selectbox("What are you rating?", list(targets), key=f"target-{run['run_id']}")
        verdict = st.selectbox(
            "Verdict",
            VERDICTS,
            format_func=str.capitalize,
            key=f"verdict-{run['run_id']}",
            help="Marking a recommendation useful lets it answer similar tickets from the cache.",
        )
        comment = st.text_input("Comment (optional)", max_chars=2000, key=f"comment-{run['run_id']}")
        submitted = st.form_submit_button("Submit feedback", key=f"submit-{run['run_id']}")
    if not submitted:
        return

    target_type, target_ref = targets[target]
    body = {
        "target_type": target_type,
        "target_ref": target_ref,
        "verdict": verdict,
        "comment": comment.strip() or None,
    }
    try:
        client.submit_feedback(run["run_id"], body)
    except ApiError as exc:
        st.error(plain(exc))
        return
    st.toast("Feedback recorded")
    st.rerun()


def show_run(client: ApiClient, run_id: str) -> None:
    """Show a run's result, waiting for it if it is still in progress."""
    try:
        run = wait_for_run(client, run_id)
    except ApiError as exc:
        st.error(plain(exc))
        return

    if run["status"] not in FINISHED_STATUSES:
        st.warning("The analysis is taking longer than expected.")
        st.button("Check again", key=f"recheck-{run_id}")
        return
    if run["status"] == "FAILED" or not run["result"]:
        st.error(plain(run["error"] or "The analysis failed."))
        return

    result = run["result"]
    banner, message = OUTCOME_BANNERS[result["outcome"]]
    banner(message)
    st.caption(f"Ticket {plain(run['ticket_id'])} · Run {plain(run['run_id'])}")

    classification = result["classification"] or {}
    confidence, priority = st.columns(2)
    confidence.metric("Confidence", percent(result["confidence"]))
    priority.metric("Priority", classification.get("priority", "n/a"))
    if classification:
        st.markdown(f"**Category:** {plain(classification['category'])} / {plain(classification['subcategory'])}")

    if result["cache_hit"]:
        st.info("Answered from the cache, using a recommendation an analyst already approved.")
    for reason in result["reasons"]:
        st.caption(plain(reason))

    show_recommendation(result)
    if result["clarifying_questions"]:
        st.subheader("Ask the reporter")
        st.markdown("\n".join(f"- {plain(question)}" for question in result["clarifying_questions"]))
    show_matches("Possible duplicates", result["duplicates"])
    show_matches("Similar cases", result["similar_cases"])
    show_details(result)
    show_feedback(client, run, feedback_targets(result))


def analyze_page(client: ApiClient) -> None:
    st.title("Analyze a ticket")
    if st.session_state.pop("refresh_ticket_id", False):
        st.session_state.ticket_id = new_ticket_id()
    st.session_state.setdefault("ticket_id", new_ticket_id())

    form_column, result_column = st.columns([2, 3], gap="large")
    with form_column:
        st.selectbox(
            "Start from a sample",
            ["", *SAMPLES],
            format_func=lambda name: name or "Blank ticket",
            key="sample",
            on_change=load_sample,
        )
        with st.form("ticket"):
            st.text_input(
                "Ticket ID",
                key="ticket_id",
                help="Must be new for each ticket. Reusing one with different text is rejected.",
            )
            st.text_input("Title", key="title", max_chars=255)
            st.text_area("Description", key="description", height=130)
            st.text_input("Symptoms (optional)", key="symptoms")
            st.text_input("Affected service", key="affected_service", max_chars=128)
            environment_column, access_column = st.columns(2)
            environment_column.selectbox(
                "Environment", ENVIRONMENTS, key="environment", format_func=lambda name: name or "Not set"
            )
            access_column.selectbox("Access level", ACCESS_LEVELS, index=1, key="access_level")
            submitted = st.form_submit_button("Analyze", type="primary", key="analyze")

        if submitted:
            state = st.session_state
            ticket = {
                "ticket_id": state.ticket_id,
                "title": state.title,
                "description": state.description,
                "symptoms": state.symptoms.strip() or None,
                "affected_service": state.affected_service,
                "environment": state.environment or None,
                "access_level": state.access_level,
            }
            try:
                accepted = client.analyze(ticket)
            except ApiError as exc:
                st.error(plain(exc))
            else:
                state.run_id = accepted["run_id"]
                state.refresh_ticket_id = True
                st.rerun()

    with result_column:
        run_id = st.session_state.get("run_id")
        if run_id:
            show_run(client, run_id)
        else:
            st.info("Submit a ticket to see its analysis here.")


def show_chat_turn(turn: dict[str, Any]) -> None:
    """Render one chat message. Replies are escaped, keeping their line breaks."""
    with st.chat_message(turn["role"]):
        st.markdown(plain(turn["content"]).replace("\n", "  \n"))
        reply = turn.get("reply")
        if reply is None:
            return
        if reply["kind"] == "NO_ANSWER":
            st.caption("No verified answer. An analyst should take this one.")
        if reply["evidence"]:
            with st.expander(f"Verified sources ({len(reply['evidence'])})"):
                for claim in reply["evidence"]:
                    st.markdown(f"**{plain(claim['source_id'])}**")
                    st.markdown("> " + plain(claim["excerpt"]).replace("\n", "\n> "))
                    st.caption(plain(claim["reason"]))
        cited = {claim["source_id"] for claim in reply["evidence"]}
        related = [match for match in reply["related"] if match["source_id"] not in cited]
        if related:
            with st.expander(f"Related cases ({len(related)})"):
                st.markdown("\n".join(f"- {plain(m['source_id'])}: {plain(m['title'])}" for m in related))


def send_chat(client: ApiClient, turns: list[dict[str, Any]]) -> None:
    """Get a reply and store it. Unanswered messages are dropped on failure, so they can be resent."""
    messages = [{"role": turn["role"], "content": turn["content"]} for turn in turns[-CHAT_HISTORY_MESSAGES:]]
    try:
        with st.chat_message("assistant"), st.spinner("Looking through past tickets..."):
            reply = client.chat(messages)
    except ApiError as exc:
        while turns and turns[-1]["role"] == "user":
            turns.pop()
        st.error(plain(exc))
        return
    turns.append({"role": "assistant", "content": reply["answer"], "reply": reply})
    st.rerun()


def chat_page(client: ApiClient) -> None:
    st.title("Chat with the assistant")
    st.caption("Describe an IT problem. Fixes come from past tickets and approved knowledge, with sources.")
    turns = st.session_state.setdefault("chat_turns", [])
    prompt = st.chat_input("Describe your IT problem", max_chars=CHAT_MESSAGE_CHARS, key="chat_prompt")
    if prompt and prompt.strip():
        turns.append({"role": "user", "content": prompt.strip()})

    if turns and st.sidebar.button("New chat", key="new_chat"):
        turns.clear()
        st.rerun()
    for turn in turns:
        show_chat_turn(turn)
    if not turns:
        for index, example in enumerate(CHAT_EXAMPLES):
            if st.button(example, key=f"chat_example_{index}"):
                turns.append({"role": "user", "content": example})
                st.rerun()
    # Sending while a reply is pending interrupts it, so this also picks up that earlier message.
    if turns and turns[-1]["role"] == "user":
        send_chat(client, turns)


def lookup_page(client: ApiClient) -> None:
    st.title("Look up a run")
    run_id = st.text_input("Run ID", key="lookup_run_id").strip()
    if run_id:
        show_run(client, run_id)


def search_page(client: ApiClient) -> None:
    st.title("Search tickets")
    with st.form("search"):
        text_column, application_column, group_column = st.columns(3)
        text_column.text_input("Title contains", key="search_q")
        application_column.text_input("Application", key="search_application")
        group_column.text_input("Assignment group", key="search_group")
        category, status, priority, environment = st.columns(4)
        category.selectbox("Category", ("", *CATEGORIES), key="search_category", format_func=lambda v: v or "Any")
        status.selectbox("Status", ("", *TICKET_STATUSES), key="search_status", format_func=lambda v: v or "Any")
        priority.selectbox("Priority", ("", *PRIORITIES), key="search_priority", format_func=lambda v: v or "Any")
        environment.selectbox(
            "Environment", ENVIRONMENTS, key="search_environment", format_func=lambda v: v or "Any"
        )
        from_column, to_column = st.columns(2)
        created_from: date | None = from_column.date_input("Created from", value=None, key="search_from")
        created_to: date | None = to_column.date_input("Created to", value=None, key="search_to")
        if st.form_submit_button("Search", type="primary", key="search_submit"):
            st.session_state.search_page_number = 1

    page = st.session_state.get("search_page_number", 1)
    try:
        results = client.search(
            q=st.session_state.search_q,
            application=st.session_state.search_application,
            assignment_group=st.session_state.search_group,
            category=st.session_state.search_category,
            status=st.session_state.search_status,
            priority=st.session_state.search_priority,
            environment=st.session_state.search_environment,
            created_from=datetime.combine(created_from, day_time.min).isoformat() if created_from else None,
            created_to=datetime.combine(created_to, day_time.max).isoformat() if created_to else None,
            limit=PAGE_SIZE,
            offset=(page - 1) * PAGE_SIZE,
        )
    except ApiError as exc:
        st.error(plain(exc))
        return

    st.caption(f"{results['total']} tickets found")
    if results["items"]:
        rows = [
            {
                "Ticket": item["ticket_id"],
                "Title": item["title"],
                "Application": item["affected_service"],
                "Env": item["environment"],
                "Category": item["category"],
                "Priority": item["priority"],
                "Status": item["status"],
                "Group": item["assignment_group"],
                "Created": item["created_at"][:10],
            }
            for item in results["items"]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    pages = max(1, -(-results["total"] // PAGE_SIZE))
    if pages > 1:
        st.number_input("Page", min_value=1, max_value=pages, step=1, key="search_page_number")


def main() -> None:
    st.set_page_config(page_title="IT Ticket Assistant", layout="wide")
    st.sidebar.title("IT Ticket Assistant")
    client = connect()
    if client is None:
        st.title("IT Ticket Assistant")
        st.info("Connect with an API key in the sidebar to get started.")
        return

    pages = {
        "Chat": chat_page,
        "Analyze a ticket": analyze_page,
        "Look up a run": lookup_page,
        "Search tickets": search_page,
    }
    choice = st.sidebar.radio("Go to", list(pages), key="page")
    st.sidebar.caption("Suggestions are for analyst review. Nothing here changes a live system.")
    try:
        pages[choice](client)
    finally:
        client.close()


main()
