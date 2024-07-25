
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import VectorDBQAChain
from langchain.llms import OpenAI
import os
import faiss
import pandas as pd
import numpy as np
import re
import random
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load local LLM
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Ensure the FAISS index and data file exist
index_path = 'faiss_mock_index'
data_path = 'indexed_mock_data.csv'

if os.path.exists(index_path) and os.path.exists(data_path):
    # Load FAISS index and data
    index = faiss.read_index(index_path)
    mock_data = pd.read_csv(data_path)
else:
    st.error("Index or data file not found. Please ensure the files are created correctly.")

# Initialize LangChain components
llm = OpenAI(temperature=0.7)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector store
vector_store = FAISS(embedding_model.embed_query, index)

# Create QA chain using VectorDBQAChain
qa_chain = VectorDBQAChain(llm=llm, vectorstore=vector_store)

def parse_days(text):
    match = re.search(r'(\d+)\s*days?', text)
    if match:
        return int(match.group(1))
    return None

def generate_itinerary(destination, days):
    activities = ["explore the city", "visit museums", "enjoy the nightlife", "shop at local markets", "take a day trip to nearby attractions"]
    itinerary = []
    for day in range(1, days + 1):
        activity = random.choice(activities)
        itinerary.append(f"Day {day}: {activity.capitalize()}.")
    return itinerary

def get_recommendations(user_input):
    days = parse_days(user_input)
    if days is None:
        days = 5  # Default to 5 days if no specific duration is mentioned
    
    destination_match = re.search(r'\b(Paris|London|New York|Tokyo|Sydney|Rome|Dubai|Barcelona|Istanbul|Amsterdam|Hong Kong|Bangkok|Singapore|Los Angeles|San Francisco)\b', user_input, re.IGNORECASE)
    if destination_match:
        destination = destination_match.group(0)
        itinerary = generate_itinerary(destination, days)
        return f"Here is a travel package for {days} days in {destination}:\n" + "\n".join(itinerary)
    
    # Perform similarity search in LangChain
    try:
        query_result = qa_chain.run({"query": user_input})
        combined_recommendations = "\n\n".join([doc.page_content for doc in query_result["documents"]])
    except Exception as e:
        return f"Error performing similarity search: {e}"
    
    return combined_recommendations

# Streamlit UI
st.title("Travel Recommendation Chatbot")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    st.write(chat)

# User input
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append(f"You: {user_input}")
        if os.path.exists(index_path) and os.path.exists(data_path):
            response = get_recommendations(user_input)
            st.session_state.chat_history.append(f"Bot: {response}")
        else:
            st.session_state.chat_history.append("Bot: Index or data file not found. Please ensure the files are created correctly.")
        st.experimental_rerun()
