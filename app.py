"""
Cloud Task Manager using Supabase (Postgres + Auth).
Run with: streamlit run app.py
Backend: no local DB, all data in Supabase.
"""

import os

import streamlit as st
from supabase import create_client, Client


# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Cloud Task Manager",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error(
        "Missing Supabase configuration. Set SUPABASE_URL and SUPABASE_ANON_KEY "
        "environment variables before running this app."
    )
    st.stop()


@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


supabase = get_supabase_client()


# -------------------- AUTH HELPERS --------------------
def set_session(session):
    st.session_state["session"] = {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
        "user": session.user,
    }


def current_user():
    sess = st.session_state.get("session")
    if not sess:
        return None
    return sess["user"]


def require_auth():
    user = current_user()
    if not user:
        st.stop()
    return user


def sign_out():
    st.session_state.pop("session", None)
    supabase.auth.sign_out()


# -------------------- AUTH UI --------------------
st.title("✓ Cloud Task Manager")
st.caption("Supabase-backed tasks • Secure, multi-user, cloud-only.")

if "session" not in st.session_state:
    tab_login, tab_signup = st.tabs(["Login", "Sign up"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            if not email or not password:
                st.warning("Enter email and password.")
            else:
                res = supabase.auth.sign_in_with_password(
                    {"email": email, "password": password}
                )
                if res.session is None:
                    st.error("Invalid credentials or unverified email.")
                else:
                    set_session(res.session)
                    st.rerun()

    with tab_signup:
        s_email = st.text_input("Email", key="signup_email")
        s_password = st.text_input(
            "Password (min 6 chars)", type="password", key="signup_password"
        )
        if st.button("Sign up"):
            if not s_email or not s_password:
                st.warning("Enter email and password.")
            else:
                res = supabase.auth.sign_up(
                    {"email": s_email, "password": s_password}
                )
                if res.user is None:
                    st.error("Could not sign up. Check email or try again.")
                else:
                    st.success(
                        "Check your email to verify your account before logging in."
                    )

    st.stop()


# -------------------- AUTHENTICATED AREA --------------------
user = require_auth()

with st.sidebar:
    st.success(f"Signed in as {user.email}")
    if st.button("Logout"):
        sign_out()
        st.rerun()


# -------------------- DATA ACCESS (Supabase) --------------------
def fetch_tasks():
    # RLS should restrict rows to auth.uid() on the backend.
    resp = (
        supabase.table("tasks")
        .select("id, title, status, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data or []


def create_task(title: str):
    title = title.strip()
    if not title:
        return
    # user_id is optional if RLS uses auth.uid(); keep for clarity.
    supabase.table("tasks").insert(
        {"title": title, "status": "Pending", "user_id": user.id}
    ).execute()


def update_task_status(task_id: str, status: str):
    supabase.table("tasks").update({"status": status}).eq("id", task_id).execute()


def delete_task(task_id: str):
    supabase.table("tasks").delete().eq("id", task_id).execute()


tasks = fetch_tasks()
total = len(tasks)
completed = sum(1 for t in tasks if t["status"] == "Completed")
pending = total - completed


# -------------------- METRICS DASHBOARD --------------------
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Tasks", total)
with m2:
    st.metric("Completed", completed)
with m3:
    st.metric("Pending", pending)

st.divider()


# -------------------- CREATE --------------------
st.subheader("Add a task")
col_input, col_btn = st.columns([4, 1])
with col_input:
    new_task = st.text_input(
        "Task name",
        placeholder="e.g. Review report, Call client…",
        label_visibility="collapsed",
        key="new_task_input",
    )
with col_btn:
    add_clicked = st.button("Add task", type="primary", use_container_width=True)

if add_clicked:
    if not new_task or not new_task.strip():
        st.warning("Enter a task name.")
    else:
        create_task(new_task)
        st.rerun()

st.divider()


# -------------------- READ --------------------
st.subheader("All tasks")
if tasks:
    df_data = [
        {
            "ID": t["id"],
            "Task": t["title"],
            "Status": t["status"],
            "Created": t["created_at"],
        }
        for t in tasks
    ]
    st.dataframe(df_data, use_container_width=True, hide_index=True)
else:
    st.info("No tasks yet. Add one above.")

st.divider()


# -------------------- UPDATE / DELETE --------------------
st.subheader("Update or delete")
if tasks:
    options = [f"{t['id']} — {t['title']} ({t['status']})" for t in tasks]
    choice = st.selectbox(
        "Choose a task",
        options=options,
        label_visibility="collapsed",
        key="task_choice",
    )
    if choice:
        task_id = choice.split(" — ")[0]
        c1, c2, c3, _ = st.columns([1, 1, 1, 3])
        with c1:
            if st.button("✓ Completed", key="complete_btn"):
                update_task_status(task_id, "Completed")
                st.rerun()
        with c2:
            if st.button("↩ Pending", key="pending_btn"):
                update_task_status(task_id, "Pending")
                st.rerun()
        with c3:
            if st.button("Delete", key="delete_btn"):
                delete_task(task_id)
                st.rerun()
else:
    st.caption("Add tasks to enable update and delete.")

