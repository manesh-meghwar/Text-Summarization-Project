import streamlit as st
from transformers import AutoTokenizer, pipeline

MODEL_PATH = "artifacts/model_trainer/pegasus-samsum-model"
TOKENIZER_PATH = "artifacts/model_trainer/tokenizer"

st.set_page_config(page_title="Text Summarizer", page_icon="📝")
st.title("Text Summarizer")
st.write("Paste a dialogue or paragraph. The fine-tuned Pegasus model returns a short summary.")

@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return pipeline("summarization", model=MODEL_PATH, tokenizer=tokenizer)

summarizer = load_summarizer()

text = st.text_area("Input text", height=220, placeholder="Paste the text to summarize...")

if st.button("Summarize"):
    if not text.strip():
        st.warning("Paste some text first.")
    else:
        with st.spinner("Summarizing..."):
            result = summarizer(
                text,
                length_penalty=0.8,
                num_beams=8,
                max_length=128,
            )[0]["summary_text"]
        st.subheader("Summary")
        st.write(result)