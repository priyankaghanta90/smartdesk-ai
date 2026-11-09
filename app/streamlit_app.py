"""Streamlit chat UI for SmartDesk.

    streamlit run app/streamlit_app.py
"""

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartdesk.agent.flow import SmartDeskAgent, TurnResult  # noqa: E402
from smartdesk.config import get_settings  # noqa: E402

st.set_page_config(page_title="SmartDesk AI", page_icon="🛟", layout="centered")


# --- sign-in --------------------------------------------------------------------
# Production: the employee's identity comes from Google sign-in, never from an email typed in the chat,
# so nobody can look up a colleague's tickets. Without sign-in configured the app runs in demo mode.
def auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:  # no secrets.toml at all
        return False


def require_sign_in() -> str | None:
    """Return the signed-in employee's email, None in demo mode, or render the sign-in screen and stop."""
    if not auth_configured():
        if get_settings().require_verified_email:
            st.title("🛟 SmartDesk AI")
            st.error("Sign-in isn't set up yet. Copy `.streamlit/secrets.toml.example` to "
                     "`.streamlit/secrets.toml` and fill in your Google OAuth client (see README → Sign-in setup).")
            st.stop()
        return None  # demo mode: SmartDesk asks for the email in the chat
    if not st.user.is_logged_in:
        st.title("🛟 SmartDesk AI")
        st.caption("NovaTech Solutions · IT & HR help desk")
        st.button("Sign in with Google", on_click=st.login, type="primary")
        st.stop()

    s = get_settings()
    email = (st.user.get("email") or "").lower()
    if not (s.allowed_email_domains or s.allowed_emails):  # fail closed: no allow-list means nobody
        st.title("🛟 SmartDesk AI")
        st.error("No accounts are allowed yet. Set `ALLOWED_EMAIL_DOMAINS` (e.g. `[\"novatech.com\"]`) or "
                 "`ALLOWED_EMAILS` (e.g. `[\"you@gmail.com\"]`) in `.env` and restart the app.")
        st.button("Sign out", on_click=st.logout)
        st.stop()
    if not email or not st.user.get("email_verified", False) or not s.email_allowed(email):
        st.title("🛟 SmartDesk AI")
        st.error(f"{email or 'This account'} isn't allowed to use SmartDesk. Please sign in with your work account.")
        st.button("Sign out", on_click=st.logout)
        st.stop()
    return email


user_email = require_sign_in()


@st.cache_resource(show_spinner="Loading knowledge base and models…")
def get_agent() -> SmartDeskAgent:
    return SmartDeskAgent()


def new_session() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = [("assistant", "Hi! I'm **SmartDesk**, NovaTech's IT & HR assistant. "
                                 "Ask me about policies or IT issues, raise a ticket, or check a ticket's status.")]
    st.session_state.last = None


# First visit, or a different account signed in on this tab (user_email is None in demo mode).
if "thread_id" not in st.session_state or st.session_state.get("user") != user_email:
    new_session()
    st.session_state.user = user_email

agent = get_agent()


def send(text: str) -> None:
    st.session_state.history.append(("user", text))
    with st.spinner("Thinking…"):
        result: TurnResult = agent.send(st.session_state.thread_id, text, user_email=user_email)
    st.session_state.history.append(("assistant", result.reply))
    st.session_state.last = result


# --- sidebar --------------------------------------------------------------------
with st.sidebar:
    st.header("Session")
    last: TurnResult | None = st.session_state.last
    if user_email:
        st.write("**Signed in as:**", user_email)
    else:
        st.write("**Email:**", (last.email if last and last.email else "not shared yet"))
    if st.button("New session", use_container_width=True):
        new_session()
        st.rerun()
    if user_email and st.button("Sign out", use_container_width=True):
        new_session()
        st.logout()

    st.header("Last answer")
    if last:
        if last.cache_hit:
            st.success("Served from semantic cache")
        if last.confidence is not None:
            st.metric("Retrieval confidence", f"{last.confidence:.2f}")
        if last.escalation_reason:
            st.warning(f"Escalated: {last.escalation_reason}")
        for s in last.sources:
            st.caption(f"[{s['ref']}] {s['title']} › {s['section']}  ·  score {s['score']}\n\n`{s['source']}`")
    else:
        st.caption("Ask something to see the retrieved sources here.")

# --- chat -----------------------------------------------------------------------
st.title("🛟 SmartDesk AI")
st.caption("NovaTech Solutions · IT & HR help desk")
if not user_email:
    st.info("**Demo mode:** Google sign-in isn't configured, so SmartDesk asks for your email in the chat and "
            "trusts it. In production, identity comes from Google sign-in (see README → Privacy).", icon="ℹ️")

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

last = st.session_state.last
if last and last.pending in ("offer_ticket", "confirm_ticket"):
    yes_label = "✅ Yes, create it" if last.pending == "confirm_ticket" else "✅ Yes, raise a ticket"
    col1, col2 = st.columns(2)
    if col1.button(yes_label, use_container_width=True):
        send("yes")
        st.rerun()
    if col2.button("✖ No, thanks", use_container_width=True):
        send("no")
        st.rerun()

if prompt := st.chat_input("Type your question…"):
    send(prompt)
    st.rerun()
