# Automated Text Summarizer Web App  
This project is an **Automated Text Summarization Web App** built using **Streamlit**. It integrates both classical NLP techniques and modern Transformer-based models to provide concise summaries of long-form text.


## Key Features

### Extractive Summarization
- Uses the **LexRank algorithm** implemented via the `sumy` library.
- Selects and extracts the most relevant sentences from the original text.
- Maintains the factual structure without rephrasing or generating new content.

### Abstractive Summarization
- Utilizes Hugging Face’s **`facebook/bart-large-cnn`** model.
- Based on an **encoder-decoder Transformer architecture**.
- Generates new, human-like summaries that capture the core meaning of the text.
- Incorporates contextual understanding, rephrasing, and restructuring for fluent summaries.




## Tech Stack

- **Streamlit** – for building the interactive web application.  
- **Hugging Face Transformers** – specifically the `facebook/bart-large-cnn` model for abstractive summarization.  
- **Sumy** – using the LexRank algorithm for extractive summarization.  
- **Python** – programming language for development and integration.

