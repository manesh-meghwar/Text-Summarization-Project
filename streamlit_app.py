import torch
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "artifacts/model_trainer/pegasus-samsum-model"
TOKENIZER_PATH = "artifacts/model_trainer/tokenizer"

LENGTH_PRESETS = {
    "Short": {
        "label": "A few sentences",
        "min_length": 8,
        "max_length": 40,
        "length_penalty": 0.5,
    },
    "Medium": {
        "label": "A short paragraph",
        "min_length": 20,
        "max_length": 80,
        "length_penalty": 0.8,
    },
    "Long": {
        "label": "A fuller recap",
        "min_length": 40,
        "max_length": 128,
        "length_penalty": 1.2,
    },
}

SAMPLE_DIALOGUE = """Designing effective AI systems requires balancing the Trade-off Triangle of Quality, Latency, and Cost. While increasing model size, 
expanding context windows, or adding multi-step agent reasoning can significantly elevate output quality, these enhancements inherently introduce higher computational 
costs and slower response times. Therefore, every architectural decision must be intentional; engineers must identify which corner of the triangle to prioritize based 
on the specific product requirements—favoring ultra-low latency for real-time chat, aggressive cost-optimization for high-volume consumer apps, or maximum reasoning 
quality for asynchronous, deep-research tasks."""

st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at top left, #1b2437 0%, #0e1117 42%); }
    .summary-box {
        min-height: 260px;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(96, 165, 250, 0.28);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: #e2e8f0;
        line-height: 1.65;
        font-size: 1.05rem;
    }
    .muted { color: #94a3b8; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    return tokenizer, model, device


def summarize_text(text, min_length, max_length, length_penalty):
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            min_length=min_length,
            max_length=max_length,
            length_penalty=length_penalty,
            num_beams=8,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).replace("<n>", " ").strip()


tokenizer, model, device = load_summarizer()

st.title("Dialogue summarizer")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("Input")
    if st.button("Load sample dialogue", use_container_width=False):
        st.session_state["input_text"] = SAMPLE_DIALOGUE

    text = st.text_area(
        "Text to summarize",
        height=260,
        placeholder="Paste a dialogue or paragraph...",
        key="input_text",
        label_visibility="collapsed",
    )
    word_count = len(text.split()) if text.strip() else 0
    st.caption(f"{word_count} words · input is truncated at 1024 tokens")

    length_choice = st.radio(
        "Summary length",
        options=list(LENGTH_PRESETS.keys()),
        index=1,
        horizontal=True,
        help="Short is a brief recap. Medium is the default. Long keeps more detail.",
    )
    preset = LENGTH_PRESETS[length_choice]
    st.caption(
        f"{preset['label']} · min {preset['min_length']} tokens · "
        f"max {preset['max_length']} tokens"
    )

    run = st.button("Summarize", type="primary", use_container_width=True)

with right:
    st.subheader("Summary")
    if run:
        if not text.strip():
            st.warning("Paste some text first.")
        else:
            with st.spinner(f"Writing a {length_choice.lower()} summary..."):
                result = summarize_text(
                    text,
                    preset["min_length"],
                    preset["max_length"],
                    preset["length_penalty"],
                )
            st.session_state["summary"] = result
            st.session_state["summary_length"] = length_choice

    summary = st.session_state.get("summary", "")
    if summary:
        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
        chosen = st.session_state.get("summary_length", length_choice)
        st.caption(f"{len(summary.split())} words · {chosen} setting")
        st.code(summary, language=None)
    else:
        st.markdown(
            '<div class="summary-box muted">The summary will appear here.</div>',
            unsafe_allow_html=True,
        )
