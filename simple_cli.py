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

def main():
    print("=" * 50)
    print("Agricultural Chatbot for South Sudan")
    print("=" * 50)
    print("This chatbot provides farming advice specific to South Sudan's agricultural conditions.")
    print("Ask questions about crops, soil, pests, weather, and farming techniques.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye!")
            break
        
        response = generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()