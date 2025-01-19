import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

# Load pre-trained model and tokenizer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
model.load_state_dict(torch.load("distilbert_sentiment_model.pt"))  # Path to your saved model
model.eval()  # Set the model to evaluation mode

# Title of the Streamlit app
st.title("Sentiment Analysis Dashboard")

# Input text box
text_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze"):
    if text_input:
        # Tokenize the input text
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # Perform inference with the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()

        # Display the result
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Class Probabilities: {probs.tolist()}")
    else:
        st.write("Please enter some text to analyze.")

# To run the Streamlit app, use the following command:
# streamlit run streamlit_app.py
