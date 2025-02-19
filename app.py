import nltk
import numpy as np
import random
import string
import bs4 as bs
import urllib.request
import re
from sentence_transformers import SentenceTransformer, util
import torch
import streamlit as st

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Fetch and parse the Wikipedia article
raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Tennis')
raw_html = raw_html.read()
article_html = bs.BeautifulSoup(raw_html, 'html.parser')
article_paragraphs = article_html.find_all('p')

# Extract and clean article text
article_text = ''
for para in article_paragraphs:
    article_text += para.text
article_text = article_text.lower()
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)

# Tokenize sentences for response generation
article_sentences = nltk.sent_tokenize(article_text)

# Initialize lemmatizer
wnlemmatizer = nltk.stem.WordNetLemmatizer()

def perform_lemmatization(tokens):
    return [wnlemmatizer.lemmatize(token) for token in tokens]

# Preprocessing for TF-IDF (not used in the enhanced version)
punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)

def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(punctuation_removal)))

# Set up greetings
greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup")
greeting_responses = ["hey", "hey hows you?", "*nods*", "hello, how you doing", "hello", "Welcome, I am good and you"]

def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)

# Load the sentence transformer model for enhanced response generation
model = SentenceTransformer('all-MiniLM-L6-v2')
article_sentence_embeddings = model.encode(article_sentences, convert_to_tensor=True)

# Define common responses for frequently asked questions
common_responses = {
    "what is tennis": "Tennis is a sport where players use a racquet to hit a ball over a net into the opponent's court...",
    "rules of tennis": "The game of tennis has specific rules regarding scoring, service, and in-play actions...",
    "roger federer": "Roger Federer is a Swiss professional tennis player known for his versatility, elegance, and numerous records..."
}

# Enhanced response generation using semantic search
def generate_response(user_input):
    # Check for common responses first
    for question, response in common_responses.items():
        if question in user_input.lower():
            return response

    # If no common response, use semantic similarity to find the best match
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, article_sentence_embeddings)
    best_match = int(cosine_scores.argmax())
    best_score = float(cosine_scores[0][best_match])

    # Threshold for relevance
    if best_score < 0.5:
        return "I am sorry, I couldn't find a good answer to that question."
    else:
        response = f"Here's what I found on that topic: {article_sentences[best_match]}"
        return response

#Streamlit app setup
st.title("TennisRoboChatbot")
st.write("""Hello, I am your friend TennisRobo.
         You can ask me any question regarding tennis:""")

#User input box
user_input=st.text_input("YouL ","")

if user_input:
    #Check if input is a greeting
    greeting_response=generate_greeting_response(user_input)
    if greeting_response:
        st.write("TennisRobo:",greeting_response)
    else:
        #Generate and display a response
        response=generate_response(user_input)
        st.write("TennisRobo:",response)