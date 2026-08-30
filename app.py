import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

ALLOWED_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "groq-gpt-oss-120b",
    "groq-gpt-oss-20b",
    "groq-qwen-3.6-27b",
    "groq-qwen-3.8-27b",
]
DEFAULT_MODEL = "groq-gpt-oss-120b"

ROOT = Path(__file__).parent
SALARY_CSV = ROOT / "data" / "salary_data.csv"
GOAL_CSV = ROOT / "data" / "city_goal_costs.csv"

st.set_page_config(page_title="AI Financial Dream Planner", layout="wide", initial_sidebar_state="collapsed")

ACCENT = "#8b5cf6"
TEXT_COLOR = "#f1f2f6"
MUTED_TEXT = "#a6adc8"
CARD_BG = "rgba(255,255,255,0.04)"
CARD_BORDER = "rgba(255,255,255,0.09)"
PILL_BG = "rgba(255,255,255,0.06)"
PLOT_COLORWAY = ["#8b5cf6", "#00d4b3", "#ff6b9d", "#ffb84d", "#4d9dff", "#a78bfa"]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    [data-testid="stSidebar"] {{ display: none; }}
    [data-testid="stHeader"] {{ display: none; }}
    [data-testid="stMainBlockContainer"], .block-container {{
        padding-top: 0 !important;
        padding-left: 24px !important;
        padding-right: 24px !important;
        max-width: 100% !important;
    }}

    .st-key-compose_box {{
        padding: 18px 20px;
        border-radius: 20px;
    }}
    .st-key-compose_box textarea {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-left: 4px !important;
    }}
    .st-key-compose_box [data-testid="stHorizontalBlock"] {{
        align-items: center;
        margin-top: 8px;
    }}
    .st-key-send_btn button {{
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        padding: 0 !important;
        background: linear-gradient(90deg, {ACCENT}, #7a73ff) !important;
        border: none !important;
        color: white !important;
        font-weight: 700;
    }}

    .st-key-navbar {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: #0e1117;
        border-bottom: 1px solid {CARD_BORDER};
        padding: 10px 24px;
        margin: 0 0 12px 0;
    }}
    .st-key-navbar [data-testid="stHorizontalBlock"] {{
        align-items: center;
    }}
    .st-key-navbar [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"] {{
        align-items: flex-end;
    }}
    .navbar-brand {{
        font-weight: 800;
        font-size: 1.25rem;
        color: {TEXT_COLOR};
        letter-spacing: -0.02em;
        white-space: nowrap;
    }}
    .navbar-brand span {{
        background: linear-gradient(90deg, {ACCENT}, #00d4b3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .navbar-title {{
        text-align: center;
        font-weight: 700;
        font-size: 2.05rem;
        color: {TEXT_COLOR};
        white-space: nowrap;
    }}

    div[role="radiogroup"] {{
        gap: 6px;
        justify-content: flex-end;
    }}
    div[role="radiogroup"] label {{
        background: {PILL_BG};
        border: 1px solid transparent;
        border-radius: 999px;
        padding: 6px 18px !important;
        transition: all 0.18s ease;
        cursor: pointer;
    }}
    div[role="radiogroup"] label:hover {{
        background: rgba(139,92,246,0.18);
        transform: translateY(-1px);
    }}
    div[role="radiogroup"] label[data-selected="true"] {{
        background: linear-gradient(90deg, {ACCENT}, #7a73ff);
        box-shadow: 0 4px 14px rgba(139,92,246,0.4);
    }}
    div[role="radiogroup"] label[data-selected="true"] p {{
        color: white !important;
        font-weight: 600;
    }}
    div[role="radiogroup"] label p {{
        color: {TEXT_COLOR} !important;
        margin: 0 !important;
    }}
    div[role="radiogroup"] label > div > div > div:first-child {{ display: none; }}

    .hero {{
        padding: 36px 8px 8px 8px;
    }}
    .hero h1 {{
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: {TEXT_COLOR};
        margin-bottom: 4px;
    }}
    .hero .accent {{
        background: linear-gradient(90deg, {ACCENT}, #00d4b3 60%, #ff6b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero p.subtitle {{
        font-size: 1.15rem;
        color: {MUTED_TEXT};
        max-width: 700px;
        line-height: 1.6;
    }}

    .card {{
        background: {CARD_BG};
        border-radius: 16px;
        padding: 22px 24px;
        border: 1px solid {CARD_BORDER};
        transition: all 0.25s ease;
        margin-bottom: 16px;
    }}
    .card:hover {{
        box-shadow: 0 12px 28px rgba(139,92,246,0.18);
        transform: translateY(-2px);
        border-color: rgba(139,92,246,0.4);
    }}
    .card h3 {{
        color: {TEXT_COLOR};
        font-weight: 700;
        margin-bottom: 6px;
    }}
    .card p {{ color: {MUTED_TEXT}; line-height: 1.55; }}

    [data-testid="stHorizontalBlock"]:has(.card) {{ align-items: stretch; }}
    [data-testid="stColumn"]:has(.card) > div {{ height: 100%; }}
    [data-testid="stColumn"] .card {{ height: 100%; box-sizing: border-box; }}

    .stButton > button, .stChatInput button {{
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(139,92,246,0.35);
    }}

    [data-testid="stChatMessage"] {{
        animation: fadeIn 0.35s ease;
        width: fit-content;
        max-width: 75%;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        margin-left: auto;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .stExpander {{
        border-radius: 12px !important;
        border: 1px solid {CARD_BORDER} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def call_api(method, path, timeout=30, **kwargs):
    try:
        resp = requests.request(method, f"{API_BASE_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, f"Could not reach the API at {API_BASE_URL}. Start it with `uvicorn api.main:app --reload` and refresh."
    except requests.exceptions.Timeout:
        return None, f"The request took longer than {timeout}s to answer (a multi-goal conversation with several tool calls can be slow). Try again, or ask about fewer goals at once."
    if resp.status_code != 200:
        return None, f"API error {resp.status_code}: {resp.text}"
    return resp.json(), None


def stream_chat_api(message, history, model_id):
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat/stream",
            json={"message": message, "history": history, "model_id": model_id},
            stream=True,
            timeout=CHAT_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        yield {"type": "fallback_local", "reply": f"Could not reach the API at {API_BASE_URL}. Start it with `uvicorn api.main:app --reload` and refresh.", "plan": None, "error_type": None, "suggested_models": None}
        return
    except requests.exceptions.Timeout:
        yield {"type": "fallback_local", "reply": f"The request took longer than {CHAT_TIMEOUT}s to answer. Try again, or ask about fewer goals at once.", "plan": None, "error_type": None, "suggested_models": None}
        return

    if resp.status_code != 200:
        yield {"type": "fallback_local", "reply": f"API error {resp.status_code}: {resp.text}", "plan": None, "error_type": None, "suggested_models": None}
        return

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        yield json.loads(line)


def render_message(role, content):
    with st.chat_message(role):
        st.markdown(content)


def style_fig(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR, family="Inter")),
        colorway=PLOT_COLORWAY,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=MUTED_TEXT),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor="#181b24", font_size=13, font_family="Inter", font_color=TEXT_COLOR),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def colored_bar_histogram(series, title, axis_title, bins=20):
    counts, edges = np.histogram(series.dropna(), bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [PLOT_COLORWAY[i % len(PLOT_COLORWAY)] for i in range(len(counts))]
    fig = go.Figure(go.Bar(x=centers, y=counts, marker_color=colors))
    fig.update_xaxes(title=axis_title)
    fig.update_yaxes(title="Count")
    return style_fig(fig, title)


CHAT_TIMEOUT = 240

if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.container(key="navbar"):
    nav_logo, nav_title, nav_links = st.columns([1, 2, 1.4])
    with nav_logo:
        st.markdown('<div class="navbar-brand">💡 <span>DreamPlanner</span></div>', unsafe_allow_html=True)
    with nav_title:
        st.markdown('<div class="navbar-title">AI Financial Dream Planner</div>', unsafe_allow_html=True)
    with nav_links:
        page = st.radio("Navigate", ["Home", "Our Analysis", "Predictor"], key="page", horizontal=True, label_visibility="collapsed")

if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Plan your <span class="accent">financial future</span>,<br>before it happens.</h1>
            <p class="subtitle">A local, free AI assistant for freshers - predicts your starting salary, projects what your
            biggest life goals will really cost, and tells you exactly how much to invest every month to get there.
            No paid APIs. No invented numbers. Every rupee comes from deterministic math, not a language model's guess.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cards = [
        ("🎯", "Predict", "A trained regression model (compared across 13 algorithms) estimates your starting monthly salary from your age, city, education, and job role - no work experience assumed, because you're a fresher."),
        ("📈", "Project", "Future costs for Marriage, Car, and Home are projected under a fixed 6% annual inflation rate, using real city-level cost data - not guesses."),
        ("✅", "Decide", "A feasibility engine shows exactly how much you're short (or ahead) each month for every goal, and recommends a broad investment category based on your timeline."),
    ]
    for col, (icon, title, text) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="card"><h3>{icon} {title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown("### How it works")
    st.markdown(
        """
        <div class="card">
        <p>
        1. <b>You describe yourself</b> - age, city, education, job role, and your goals (marriage, car, home) with timelines, in plain English or a structured form.<br><br>
        2. <b>A machine-learning model</b> predicts your likely starting salary - trained and compared across 13 regression algorithms on real salary data, selected by lowest error, never using "experience" as a feature.<br><br>
        3. <b>Deterministic financial tools</b> - never the AI itself - calculate future goal costs under inflation, the monthly investment each goal needs, and whether your planned savings rate is actually enough.<br><br>
        4. <b>An AI agent</b> (your choice of Gemini or Groq models) understands natural language, calls the right tools in the right order, and can research real-world investment options via live web search - but it never invents a number. Every figure traces back to a tested Python function.<br><br>
        5. <b>A local knowledge base</b> answers "what does this mean?" questions with grounded, retrieved facts - and honestly says "I don't know" rather than making something up.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Under the hood")
    tech_cols = st.columns(4)
    stack = [
        ("Machine Learning", "scikit-learn, 13 algorithms compared, MAE-selected"),
        ("Agent & LLM", "LangGraph + Gemini / Groq, tool-calling, streaming"),
        ("RAG", "Chroma + BM25 hybrid retrieval, cross-encoder reranked"),
        ("Backend", "FastAPI - the single gateway for every service"),
    ]
    for col, (label, val) in zip(tech_cols, stack):
        with col:
            st.markdown(f'<div class="card"><h3 style="font-size:0.95rem">{label}</h3><p style="font-size:0.9rem">{val}</p></div>', unsafe_allow_html=True)

    st.caption("Educational simulation only - not professional financial advice, and no returns are guaranteed.")

elif page == "Our Analysis":
    st.markdown('<div class="hero" style="padding-top:12px"><h1 style="font-size:2rem">Our Analysis</h1><p class="subtitle">Every chart below is interactive - hover for exact values, zoom by dragging, and use the camera icon in the top-right of each chart to copy/download it as an image.</p></div>', unsafe_allow_html=True)

    salary_df = pd.read_csv(SALARY_CSV)
    salary_df = salary_df.loc[:, ~salary_df.columns.str.startswith("Unnamed")]
    goal_df = pd.read_csv(GOAL_CSV)

    st.subheader("Salary Data Summary")
    st.dataframe(salary_df.describe(include="all").astype(str), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        fig = colored_bar_histogram(salary_df["Monthly_Salary"], "Monthly Salary Distribution", "Monthly Salary")
        st.plotly_chart(fig, width="stretch")
    with col2:
        order = salary_df.groupby("Education")["Monthly_Salary"].median().sort_values(ascending=False).index
        fig = px.box(salary_df, x="Education", y="Monthly_Salary", color="Education", category_orders={"Education": list(order)})
        st.plotly_chart(style_fig(fig, "Monthly Salary by Education"), width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        order = salary_df.groupby("City")["Monthly_Salary"].median().sort_values(ascending=False).index
        fig = px.box(salary_df, x="City", y="Monthly_Salary", color="City", category_orders={"City": list(order)})
        st.plotly_chart(style_fig(fig, "Monthly Salary by City"), width="stretch")
    with col4:
        pivot = salary_df.pivot_table(index="City", columns="Education", values="Monthly_Salary", aggfunc="mean")
        fig = px.imshow(pivot, text_auto=".0f", color_continuous_scale=["#181b24", ACCENT])
        st.plotly_chart(style_fig(fig, "Mean Salary by City and Education"), width="stretch")

    st.subheader("City Goal Costs")
    cost_cols = ["Marriage_Cost_Current", "Car_Cost_Current", "Home_Cost_Current"]
    cost_tabs = st.tabs(cost_cols)
    for tab, col in zip(cost_tabs, cost_cols):
        with tab:
            label = col.replace("_", " ")
            fig = colored_bar_histogram(goal_df[col], label, label)
            st.plotly_chart(fig, width="stretch")

    st.subheader("Model Training & Selection")
    metadata, error = call_api("GET", "/models/metadata")
    if error:
        st.warning(error)
    elif metadata:
        st.markdown(f"**Selected model:** `{metadata['selected_model']}` &nbsp; ({metadata['selection_rule']})")
        metrics_df = pd.DataFrame(metadata["metrics"]).T.sort_values("Test_MAE")
        st.dataframe(metrics_df, width="stretch")

        fig = px.bar(metrics_df, x=metrics_df.index, y="Test_MAE", color=metrics_df.index)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, "Test MAE by Algorithm (lower is better)"), width="stretch")

elif page == "Predictor":
    st.markdown('<div class="hero" style="padding-top:12px"><h1 style="font-size:2rem">Predictor</h1><p class="subtitle">Chat naturally. If any required field is missing (city, education, job role, age, goals, savings %), the agent will ask for it.</p></div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_notice" not in st.session_state:
        st.session_state.last_notice = None
    if "pending" not in st.session_state:
        st.session_state.pending = False
    if "model_id" not in st.session_state:
        st.session_state.model_id = DEFAULT_MODEL

    has_history = bool(st.session_state.chat_history)
    user_message = None

    if not has_history:
        st.markdown("<div style='height: 10vh'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center'>Tell me about yourself and your goals</h4>", unsafe_allow_html=True)
        _, box_col, _ = st.columns([1, 3, 1])
        with box_col:
            with st.container(key="compose_box", border=True):
                message_draft = st.text_area(
                    "Message", key="compose_text",
                    placeholder="e.g. I am 24, live in Delhi, B.Tech, Software Engineer...",
                    height=100, disabled=st.session_state.pending, label_visibility="collapsed",
                )
                _, select_col, send_col = st.columns([5, 3, 1])
                with select_col:
                    model_id = st.selectbox(
                        "Model", ALLOWED_MODELS, key="model_id",
                        disabled=st.session_state.pending, label_visibility="collapsed",
                    )
                with send_col:
                    send_clicked = st.button("↑", key="send_btn", disabled=st.session_state.pending, width="stretch")
                if send_clicked and message_draft.strip():
                    user_message = message_draft.strip()
    else:
        model_id = st.session_state.model_id

        for msg in st.session_state.chat_history:
            render_message(msg["role"], msg["content"])

        if st.session_state.last_notice:
            st.warning(st.session_state.last_notice)
            st.session_state.last_notice = None
        if st.session_state.get("last_plan"):
            with st.expander("Structured plan (JSON)"):
                st.json(st.session_state.last_plan)
            st.session_state.last_plan = None

        user_message = st.chat_input(
            "Tell me about yourself and your goals...",
            key="chat_input_active",
            disabled=st.session_state.pending,
        )

        if st.session_state.pending:
            last_message = st.session_state.chat_history[-1]["content"]
            history_for_api = st.session_state.chat_history[:-1]
            final_holder = {}

            def gen():
                for event in stream_chat_api(last_message, history_for_api, model_id):
                    if event.get("type") == "token":
                        yield event["text"]
                    else:
                        final_holder["event"] = event

            with st.chat_message("assistant"):
                full_reply = st.write_stream(gen())

            final_event = final_holder.get("event")
            reply_text = full_reply
            if final_event:
                if final_event.get("type") in ("fallback", "fallback_local"):
                    reply_text = final_event.get("reply", full_reply)
                    if final_event.get("error_type") == "rate_limit":
                        st.session_state.last_notice = f"'{model_id}' hit its rate limit. Try: {', '.join(final_event['suggested_models'])}"
                if final_event.get("plan"):
                    st.session_state.last_plan = final_event["plan"]

            st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
            st.session_state.pending = False
            st.rerun()

    if user_message and not st.session_state.pending:
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        st.session_state.pending = True
        st.rerun()
