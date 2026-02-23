import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib
from PyPDF2 import PdfReader
from PIL import Image
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from transformers import BlipProcessor, BlipForQuestionAnswering

# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")
USERS_FILE = "users.json"

# -------------------- HELPERS --------------------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

def type_text(text, speed=0.03):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(f"### {out}")
        time.sleep(speed)

# -------------------- CACHED MODELS --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@st.cache_resource
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(device)
    return processor, model, device

# -------------------- SESSION DEFAULTS --------------------
defaults = {
    "authenticated": False,
    "users": load_users(),
    "vector_db": None,
    "chat_history": [],
    "current_pdf_id": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- AUTH UI --------------------
def login_ui():
    col1, col2 = st.columns(2)

    with col1:
        st_lottie(
            load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"),
            height=300
        )

    with col2:
        type_text("🔐 Welcome to SlideSense")

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                if u in st.session_state.users and st.session_state.users[u] == hash_password(p):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab2:
            nu = st.text_input("New Username")
            np = st.text_input("New Password", type="password")
            if st.button("Create Account"):
                if nu in st.session_state.users:
                    st.warning("User already exists")
                else:
                    st.session_state.users[nu] = hash_password(np)
                    save_users(st.session_state.users)
                    st.success("Account created")

# -------------------- IMAGE Q&A --------------------
def answer_image_question(image, question):
    processor, model, device = load_blip()
    inputs = processor(image, question, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_length=10, num_beams=5)
    short_answer = processor.decode(outputs[0], skip_special_tokens=True)

    llm = load_llm()
    prompt = f"""
Question: {question}
Vision Answer: {short_answer}
Convert into one clear sentence. No extra details.
"""
    return llm.invoke(prompt).content

# -------------------- AUTH CHECK --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.success("Logged in ✅")

if st.sidebar.button("Logout"):
    st.cache_resource.clear()
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

# -------------------- SIDEBAR HISTORY --------------------
st.sidebar.markdown("### 💬 Chat History")

if st.session_state.chat_history:
    for i, (q, _) in enumerate(st.session_state.chat_history[-5:], start=1):
        st.sidebar.markdown(f"{i}. {q[:40]}...")

    if st.sidebar.button("🧹 Clear History"):
        st.session_state.chat_history = []
        st.rerun()
else:
    st.sidebar.caption("No history yet")

# -------------------- HERO --------------------
col1, col2 = st.columns([1, 2])

with col1:
    st_lottie(
        load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"),
        height=250
    )

with col2:
    type_text("📘 SlideSense AI Platform")
    st.markdown("### Smart Learning | Smart Vision | Smart AI")

st.divider()

# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":
    pdf = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id
            st.session_state.vector_db = None
            st.session_state.chat_history = []

        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""

                for pdf_page in reader.pages:
                    extracted = pdf_page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                if not text.strip():
                    st.error("No readable text found in PDF")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=80
                )
                chunks = splitter.split_text(text)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        q = st.text_input("Ask a question")

        if q:
            llm = load_llm()
            docs = st.session_state.vector_db.similarity_search(q, k=5)

            prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in the document
""")

            chain = create_stuff_documents_chain(llm, prompt)
            res = chain.invoke({"context": docs, "question": q})

            if isinstance(res, dict):
                answer = res.get("output_text", "")
            else:
                answer = res

            st.session_state.chat_history.append((q, answer))

        # -------- CHAT DISPLAY (QUESTION ON TOP, ANSWER BELOW) --------
        st.markdown("## 💬 Conversation")

        chat_container = st.container()
        with chat_container:
            for uq, ua in st.session_state.chat_history:
                st.markdown(f"🧑 **You:** {uq}")
                st.markdown(f"🤖 **AI:** {ua}")
                st.divider()

# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":
    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)

        question = st.text_input("Ask a question about the image")
        if question:
            with st.spinner("Analyzing image..."):
                st.success(answer_image_question(img, question))

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

