# Dialogue Summarizer

A Streamlit app that summarizes short chat-style conversations using a
Flan-T5-base model fine-tuned with LoRA on the SAMSum dataset.

**Live app:** [(https://dialogue-summarizer-app-yhg4rto4v4ytsdqyjcfvkb.streamlit.app/)]
**Model:** [Sasmita03/lora-samsum-summarizer](https://huggingface.co/Sasmita03/lora-samsum-summarizer)

## How it works
Paste a conversation into the text box and click Summarize. The app loads
the base Flan-T5 model and attaches the LoRA adapter at runtime to generate
a one- or two-sentence summary.
