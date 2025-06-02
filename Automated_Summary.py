import streamlit as st
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


#Load abstractive summarization model
@st.cache_resource
def load_abstractive_model():
    return pipeline("summarization",model="facebook/bart-large-cnn",framework="pt",tokenizer="facebook/bart-large-cnn")

summarizer_abstractive=load_abstractive_model()


#Extractive summarization function using sumy

def extractive_summary(text,num_sentences=3):
    parser=PlaintextParser.from_string(text,Tokenizer("english"))
    summarizer=LexRankSummarizer()
    summary=summarizer(parser.document,num_sentences)
    return " ".join([str(sentence) for sentence in summary])


#Abstractive summary using Huggingface pipeline

def abstractive_summary(text):
    max_input_len=1000
    if len(text.split*())>max_input_len:
        text=" ".join(text.split()[:max_input_len])

    result=summarizer_abstractive(text,max_length=150,min_length=30,do_sample=False)
    return result[0]['summary_text']


#Streamlit UI

st.set_page_config(page_title="Text Summarization WebApp",layout="centered")

st.st.title("📝 Automated Text Summarization")
st.write("Enter text below and choose extractive or abstractive summarization.")

text_input = st.text_area("📄 Input Text", height=300)

if st.button("Generate Summaries"):
    if text_input.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Generating summaries..."):
            extractive = extractive_summary(text_input)
            abstractive = abstractive_summary(text_input)

        st.subheader("🔍 Extractive Summary")
        st.write(extractive)

        st.subheader("🧠 Abstractive Summary")
        st.write(abstractive)
