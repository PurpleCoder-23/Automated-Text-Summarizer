import streamlit as st
import nltk
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Download required NLTK tokenizer
nltk.download('punkt')
nltk.download('punkt_tab') # This download is typically not needed for sumy/transformers and might cause issues if not available.
                               # 'punkt' is usually sufficient for tokenization.

# Set Streamlit page config
st.set_page_config(page_title="Text Summarization WebApp", layout="wide") # Changed layout to 'wide' for more space

# Load abstractive summarization model
@st.cache_resource
def load_abstractive_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        framework="pt",
        tokenizer="facebook/bart-large-cnn"
    )

summarizer_abstractive = load_abstractive_model()

# Extractive summarization function using sumy
def extractive_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# Abstractive summary using Hugging Face pipeline
def abstractive_summary(text):
    max_input_len = 500  # Reduced for demonstration, adjust as needed
    original_text_len = len(text.split())

    if original_text_len > max_input_len:
        text = " ".join(text.split()[:max_input_len])
        st.info(f"Input text truncated for abstractive summary from {original_text_len} words to {max_input_len} words.")

    result = summarizer_abstractive(text, max_length=150, min_length=30, do_sample=False)
    return result[0]['summary_text']

# Streamlit UI
st.title("📝 Automated Text Summarization")
st.write("Enter text below and choose extractive or abstractive summarization.")

# Input text area in the main column
text_input = st.text_area("📄 Input Text", height=150)

# Move the num_sentences slider to the sidebar
with st.sidebar:
    st.header("Settings")
    num_sentences_extractive = st.slider(
        "Number of sentences for Extractive Summary",
        min_value=1,
        max_value=10,
        value=3,
        key="num_sentences_slider" # Added a key to prevent potential issues
    )

if st.button("Generate Summaries"):
    if text_input.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Generating summaries..."):
            extractive = extractive_summary(text_input, num_sentences=num_sentences_extractive)
            abstractive = abstractive_summary(text_input)

        # Use columns to display summaries side-by-side
        col1, col2 = st.columns(2) # Create two columns

        with col1:
            st.subheader("🔍 Extractive Summary")
            st.write(extractive)

        with col2:
            st.subheader("🧠 Abstractive Summary")
            st.write(abstractive)