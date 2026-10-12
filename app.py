import re
import time

import streamlit as st

from pipeline import STEPS, run_research_steps

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🔬",
    layout="wide",
)

# ── Styling (works in light and dark theme) ───────────────────────────────────
st.markdown(
    """
<style>
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1100px; padding-top: 2.5rem; }

.step-row {
    display: flex; align-items: center; gap: .9rem;
    padding: .8rem 1rem; margin-bottom: .55rem;
    border: 1px solid rgba(128,128,128,.28); border-radius: 12px;
}
.step-row.running { border-color: #ff8c32; background: rgba(255,140,50,.08); }
.step-row.done    { border-color: rgba(60,180,100,.55); background: rgba(60,180,100,.07); }
.step-row.failed  { border-color: #e5484d; background: rgba(229,72,77,.08); }
.step-icon  { font-size: 1.15rem; width: 1.6rem; text-align: center; }
.step-title { font-weight: 650; }
.step-desc  { font-size: .82rem; opacity: .65; }
.step-state { margin-left: auto; font-size: .75rem; letter-spacing: .08em; opacity: .75; }
.step-row.running .step-state { color: #ff8c32; opacity: 1; }
.step-row.done .step-state    { color: #3cb464; opacity: 1; }
.step-row.failed .step-state  { color: #e5484d; opacity: 1; }
</style>
""",
    unsafe_allow_html=True,
)

EXAMPLES = ["LLM agents in 2025", "CRISPR gene editing", "Fusion energy progress"]

ICONS = {"waiting": "⚪", "running": "🔄", "done": "✅", "failed": "❌"}
LABELS = {"waiting": "WAITING", "running": "RUNNING", "done": "DONE", "failed": "FAILED"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def render_tracker(placeholder, states):
    """Draw the four pipeline steps. Only static labels go into the HTML."""
    rows = []
    for key, title, desc in STEPS:
        s = states[key]
        rows.append(
            f'<div class="step-row {s}">'
            f'<span class="step-icon">{ICONS[s]}</span>'
            f'<div><div class="step-title">{title}</div>'
            f'<div class="step-desc">{desc}</div></div>'
            f'<span class="step-state">{LABELS[s]}</span></div>'
        )
    placeholder.markdown("".join(rows), unsafe_allow_html=True)


def parse_score(feedback: str):
    m = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback, re.IGNORECASE)
    return m.group(1) if m else None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:50] or "report"


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "429" in msg or "quota" in low or "resource_exhausted" in low or "rate limit" in low:
        return ("The AI model hit its free-tier rate limit. "
                "Wait a minute and run it again.\n\n" + msg[:400])
    if "api key" in low or "api_key" in low or "permission" in low or "401" in msg:
        return "There's a problem with an API key. Check your `.env` file.\n\n" + msg[:400]
    return msg[:600]


def set_example(text: str):
    st.session_state.topic = text


def run_pipeline(topic: str):
    """Run the pipeline, updating the step tracker live; store the outcome."""
    states = {key: "waiting" for key, _, _ in STEPS}
    outputs = {}
    started = time.time()

    tracker = st.empty()
    render_tracker(tracker, states)

    try:
        for event in run_research_steps(topic):
            states[event["step"]] = event["status"]
            if event["status"] == "done":
                outputs[event["key"]] = event["output"]
            render_tracker(tracker, states)
    except Exception as exc:  # show the failure instead of a raw traceback
        for key, s in states.items():
            if s == "running":
                states[key] = "failed"
        render_tracker(tracker, states)
        st.session_state.error = friendly_error(exc)
        st.session_state.result = None
        return

    st.session_state.error = None
    st.session_state.result = {
        "topic": topic,
        "outputs": outputs,
        "seconds": time.time() - started,
    }


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in (("topic", ""), ("result", None), ("error", None)):
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 ResearchMind")
st.caption(
    "Four AI steps work together: search the web, read a source, write a report, "
    "then critique it."
)

# ── Input ─────────────────────────────────────────────────────────────────────
topic = st.text_input(
    "Research topic",
    key="topic",
    placeholder="e.g. Quantum computing breakthroughs in 2025",
)
run_clicked = st.button("⚡ Run research", type="primary", use_container_width=True)

st.caption("Or try an example:")
example_cols = st.columns(len(EXAMPLES))
for col, example in zip(example_cols, EXAMPLES):
    col.button(example, key=f"ex_{example}", on_click=set_example, args=(example,),
               use_container_width=True)

# ── Run ───────────────────────────────────────────────────────────────────────
if run_clicked:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.result = None
        st.session_state.error = None
        st.divider()
        st.subheader("Progress")
        run_pipeline(topic.strip())

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(st.session_state.error)

result = st.session_state.result
if result:
    out = result["outputs"]
    report = out.get("report", "")
    feedback = out.get("feedback", "")
    score = parse_score(feedback)

    st.divider()
    head_left, head_mid, head_right = st.columns([3, 1, 1])
    head_left.subheader(f"Results: {result['topic']}")
    head_mid.metric("Critic score", f"{score}/10" if score else "n/a")
    head_right.metric("Time taken", f"{result['seconds']:.0f}s")

    tab_report, tab_critic, tab_raw = st.tabs(
        ["📝 Report", "🧐 Critic feedback", "🔎 Raw research"]
    )

    with tab_report:
        st.markdown(report)
        st.download_button(
            "⬇ Download report (.md)",
            data=report,
            file_name=f"research_{slugify(result['topic'])}.md",
            mime="text/markdown",
        )

    with tab_critic:
        st.markdown(feedback)

    with tab_raw:
        with st.expander("Search Agent output", expanded=False):
            st.markdown(out.get("search_results", ""))
        with st.expander("Reader Agent output (scraped content)", expanded=False):
            st.markdown(out.get("scraped_content", ""))