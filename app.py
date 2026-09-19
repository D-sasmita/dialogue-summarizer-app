import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# Change this if you fine-tune a different model later
ADAPTER_REPO = "Sasmita03/lora-samsum-summarizer"
BASE_MODEL = "google/flan-t5-base"
PREFIX = "Summarize this conversation: "

st.set_page_config(page_title="Dialogue Summarizer", page_icon="💬")


@st.cache_resource(show_spinner="Loading model... (first load takes ~30s)")
def load_model():
    # Load the base model once, then attach the LoRA adapter on top.
    # @st.cache_resource means this only runs once per app session,
    # not on every button click.
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
    return model, tokenizer


st.title("💬 Dialogue Summarizer")
st.caption("Flan-T5-base fine-tuned with LoRA on the SAMSum dataset")

model, tokenizer = load_model()

example_dialogue = """Amanda: I baked cookies. Do you want some?
Jerry: Sure! Bring me some tomorrow.
Amanda: I will leave you some in the fridge.
Jerry: You're the best!"""

dialogue_input = st.text_area(
    "Paste a conversation to summarize",
    value=example_dialogue,
    height=200,
    help="Works best on short, chat-style conversations (2-10 people, a few lines each).",
)

if st.button("Summarize", type="primary"):
    if not dialogue_input.strip():
        st.warning("Paste a conversation first.")
    else:
        with st.spinner("Generating summary..."):
            inputs = tokenizer(
                PREFIX + dialogue_input,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            output = model.generate(**inputs, max_length=100)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)

        st.subheader("Summary")
        st.write(summary)

st.divider()
st.caption(
    "Built with LoRA (PEFT) on Flan-T5-base. "
    f"Adapter: [{ADAPTER_REPO}](https://huggingface.co/{ADAPTER_REPO})"
)