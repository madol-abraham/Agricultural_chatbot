import argparse
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class AgricultureChatbot:
    def __init__(self, model_path='fine_tuned_gpt2_chatbot'):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = TFGPT2LMHeadModel.from_pretrained(model_path)
        except:
            # Fallback to the base model if fine-tuned model isn't available
            print("Fine-tuned model not found. Using base GPT-2 model instead.")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = TFGPT2LMHeadModel.from_pretrained('gpt2')
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context = []
        self.max_context_length = 3  # Number of previous exchanges to remember

    def preprocess_input(self, text):
        # Simple preprocessing similar to training
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
            
            return ' '.join(tokens)
        except:
            # Fallback if tokenization fails
            return text.lower()

    def update_context(self, user_input, bot_response):
        self.context.append((user_input, bot_response))
        if len(self.context) > self.max_context_length:
            self.context.pop(0)

    def generate_response(self, input_text):
        # Check if input is agricultural related
        agri_keywords = ['crop', 'farm', 'plant', 'harvest', 'soil', 'seed', 'water', 
                        'fertilizer', 'pest', 'disease', 'agriculture', 'grow', 'field',
                        'vegetable', 'fruit', 'irrigation', 'weather', 'season', 'rain',
                        'drought', 'yield', 'organic', 'compost', 'manure', 'livestock']
        
        is_agri_related = any(keyword in input_text.lower() for keyword in agri_keywords)
        
        if not is_agri_related:
            return "I'm an agricultural chatbot. Please ask me questions related to farming, crops, soil, or other agricultural topics."
        
        # Preprocess input
        processed_input = self.preprocess_input(input_text)

        # Add context if available
        if self.context:
            context_text = " ".join([f"User: {q} Bot: {a}" for q, a in self.context])
            full_input = f"{context_text} User: {processed_input} Bot:"
        else:
            full_input = f"User: {processed_input} Bot:"

        # Tokenize and generate response
        input_ids = self.tokenizer.encode(full_input, return_tensors='tf')
        attention_mask = tf.ones_like(input_ids)

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=200,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )

        # Decode and clean response
        full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = full_response[len(full_input):].split("User:")[0].strip()

        # Update context
        self.update_context(input_text, response)

        return response

def main():
    parser = argparse.ArgumentParser(description='Agricultural Chatbot for South Sudan')
    parser.add_argument('--model_path', type=str, default='fine_tuned_gpt2_chatbot',
                        help='Path to the fine-tuned model')
    args = parser.parse_args()
    
    print("=" * 50)
    print("Agricultural Chatbot for South Sudan")
    print("=" * 50)
    print("This chatbot provides farming advice specific to South Sudan's agricultural conditions.")
    print("Ask questions about crops, soil, pests, weather, and farming techniques.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 50)
    
    chatbot = AgricultureChatbot(model_path=args.model_path)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye!")
            break
        
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()