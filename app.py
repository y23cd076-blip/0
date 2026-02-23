"""
Cloud Task Manager using Supabase (Postgres + Auth) via HTTP.
Run with: streamlit run app.py
Backend: no local DB, all data in Supabase.
"""

import os
from typing import Any, Dict, List, Optional

import requests

import streamlit as st


# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Cloud Task Manager",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error(
        "Missing Supabase configuration. Set SUPABASE_URL and SUPABASE_ANON_KEY "
        "environment variables before running this app."
    )
    st.stop()


# -------------------- LOW-LEVEL HTTP HELPERS --------------------
def auth_headers(access_token: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


def auth_request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{SUPABASE_URL}{path}"
    resp = requests.request(method, url, headers=auth_headers(), timeout=10, **kwargs)
    return resp


def db_request(
    method: str,
    path: str,
    access_token: str,
    **kwargs,
) -> requests.Response:
    url = f"{SUPABASE_URL}{path}"
    resp = requests.request(
        method,
        url,
        headers=auth_headers(access_token),
        timeout=10,
        **kwargs,
    )
    return resp


# -------------------- AUTH HELPERS --------------------
def set_session(access_token: str, refresh_token: str, user: Dict[str, Any]) -> None:
    st.session_state["session"] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": user,
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


def current_user() -> Optional[Dict[str, Any]]:
    sess = st.session_state.get("session")
    if not sess:
        return None
    return sess["user"]


def current_token() -> Optional[str]:
    sess = st.session_state.get("session")
    if not sess:
        return None
    return sess["access_token"]


def require_auth() -> Dict[str, Any]:
    user = current_user()
    if not user:
        st.stop()
    return user


def sign_out() -> None:
    st.session_state.pop("session", None)


def sign_up(email: str, password: str) -> Optional[str]:
    payload = {"email": email, "password": password}
    resp = auth_request("POST", "/auth/v1/signup", json=payload)
    if resp.status_code >= 400:
        return resp.text
    return None


def sign_in(email: str, password: str) -> Optional[str]:
    payload = {"email": email, "password": password}
    resp = auth_request(
        "POST",
        "/auth/v1/token?grant_type=password",
        json=payload,
    )
    if resp.status_code >= 400:
        return resp.text

    data = resp.json()
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    user = data.get("user")
    if not access or not user:
        return "Invalid auth response from Supabase."

    set_session(access, refresh, user)
    return None


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
                err = sign_in(email, password)
                if err:
                    st.error(f"Login failed: {err}")
                else:
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
                err = sign_up(s_email, s_password)
                if err:
                    st.error(f"Sign-up failed: {err}")
                else:
                    st.success(
                        "Check your email to verify your account before logging in."
                    )

    st.stop()


# -------------------- AUTHENTICATED AREA --------------------
user = require_auth()

with st.sidebar:
    st.success(f"Signed in as {user.get('email')}")
    if st.button("Logout"):
        sign_out()
        st.rerun()


# -------------------- DATA ACCESS (Supabase REST) --------------------
def fetch_tasks() -> List[Dict[str, Any]]:
    token = current_token()
    if not token:
        return []
    resp = db_request(
        "GET",
        "/rest/v1/tasks?select=id,title,status,created_at&order=created_at.desc",
        access_token=token,
    )
    if resp.status_code >= 400:
        st.error(f"Error loading tasks: {resp.text}")
        return []
    return resp.json()


def create_task(title: str) -> None:
    title = title.strip()
    if not title:
        return
    token = current_token()
    if not token:
        return
    payload = {"title": title, "status": "Pending", "user_id": user["id"]}
    resp = db_request(
        "POST",
        "/rest/v1/tasks",
        access_token=token,
        json=payload,
    )
    if resp.status_code >= 400:
        st.error(f"Error creating task: {resp.text}")


def update_task_status(task_id: str, status: str) -> None:
    token = current_token()
    if not token:
        return
    resp = db_request(
        "PATCH",
        f"/rest/v1/tasks?id=eq.{task_id}",
        access_token=token,
        json={"status": status},
    )
    if resp.status_code >= 400:
        st.error(f"Error updating task: {resp.text}")


def delete_task(task_id: str) -> None:
    token = current_token()
    if not token:
        return
    resp = db_request(
        "DELETE",
        f"/rest/v1/tasks?id=eq.{task_id}",
        access_token=token,
    )
    if resp.status_code >= 400:
        st.error(f"Error deleting task: {resp.text}")


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

