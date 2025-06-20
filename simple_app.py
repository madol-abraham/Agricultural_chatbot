import streamlit as st
import pandas as pd
import re
import random

# Load dataset for demo responses
try:
    df = pd.read_csv('agribusiness_qns.csv')
    qa_pairs = dict(zip(df['question'].str.lower(), df['answer']))
except:
    # Fallback data if file can't be loaded
    qa_pairs = {
        "what are the best crops to grow in south sudan": "Maize, sorghum, groundnuts, and sesame are good crops for the rainy season in South Sudan.",
        "how can i improve soil fertility": "Use organic compost, rotate crops, and apply animal manure for better soil health.",
        "what fertilizer is good for maize": "DAP at planting and UREA during top dressing stage work well for maize.",
        "how do i protect my crops from armyworms": "Use neem-based pesticides, early planting, and regular field inspection to control armyworms."
    }

# Streamlit UI
st.title("Agricultural Chatbot for South Sudan")
st.markdown("""
This chatbot provides farming advice specific to South Sudan's agricultural conditions.
Ask questions about crops, soil, pests, weather, and farming techniques.
""")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Simple response generation function
def generate_response(input_text):
    # Convert to lowercase and remove special characters
    processed_input = input_text.lower()
    processed_input = re.sub(r'[^a-zA-Z\s]', '', processed_input)
    
    # Check if input is agricultural related
    agri_keywords = ['crop', 'farm', 'plant', 'harvest', 'soil', 'seed', 'water', 
                    'fertilizer', 'pest', 'disease', 'agriculture', 'grow', 'field',
                    'vegetable', 'fruit', 'irrigation', 'weather', 'season', 'rain',
                    'drought', 'yield', 'organic', 'compost', 'manure', 'livestock']
    
    is_agri_related = any(keyword in processed_input for keyword in agri_keywords)
    
    if not is_agri_related:
        return "I'm an agricultural chatbot. Please ask me questions related to farming, crops, soil, or other agricultural topics."
    
    # Look for exact matches in our dataset
    for question, answer in qa_pairs.items():
        if processed_input in question.lower():
            return answer
    
    # If no exact match, look for partial matches
    for question, answer in qa_pairs.items():
        if any(word in question.lower() for word in processed_input.split()):
            return answer
    
    # Default response if no match found
    return "I don't have specific information about that. Please ask about crops, soil fertility, pest control, or other farming topics in South Sudan."

# Chat input
user_input = st.text_input("Ask a farming question:", key="user_input")

if st.button("Send") or user_input:
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get chatbot response
        response = generate_response(user_input)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "bot", "content": response})
        
        # Clear input
        st.session_state.user_input = ""

# Display chat history
st.subheader("Conversation")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")

# Add information about the model
st.sidebar.title("About")
st.sidebar.info("""
This chatbot provides agricultural knowledge for South Sudan.
It can answer questions about:
- Crop selection and planting times
- Soil fertility and management
- Pest and disease control
- Water management
- Post-harvest techniques
- Market access
""")