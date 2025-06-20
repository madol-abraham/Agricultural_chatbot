import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load dataset
df = pd.read_csv('agribusiness_qns.csv')
print(f"Dataset loaded with {len(df)} Q&A pairs")

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers (keep basic punctuation)
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

# Apply preprocessing
df['processed_question'] = df['question'].apply(preprocess_text)
df['processed_answer'] = df['answer'].apply(preprocess_text)

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_data(texts, max_length=128):
    return tokenizer(
        texts.tolist(),
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

# Tokenize questions and answers
questions_tokenized = tokenize_data(df['processed_question'])
answers_tokenized = tokenize_data(df['processed_answer'])

# Prepare input and target sequences
input_ids = questions_tokenized['input_ids']
attention_mask = questions_tokenized['attention_mask']
labels = answers_tokenized['input_ids']

# Split data
from sklearn.model_selection import train_test_split

# Convert TensorFlow tensors to NumPy arrays before splitting
input_ids_np = input_ids.numpy()
labels_np = labels.numpy()
attention_mask_np = attention_mask.numpy()

# Split the data
X_train, X_val, y_train, y_val, attn_train, attn_val = train_test_split(
    input_ids_np, labels_np, attention_mask_np, test_size=0.2, random_state=42
)

# Convert back to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train)
X_val = tf.convert_to_tensor(X_val)
y_train = tf.convert_to_tensor(y_train)
y_val = tf.convert_to_tensor(y_val)
attn_train = tf.convert_to_tensor(attn_train)
attn_val = tf.convert_to_tensor(attn_val)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Try to load the fine-tuned model
try:
    model = TFGPT2LMHeadModel.from_pretrained('fine_tuned_gpt2_chatbot')
    print("Loaded fine-tuned model")
except:
    # If model doesn't exist, load the base model
    print("Fine-tuned model not found. Loading base GPT-2 model.")
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Function to generate responses
def generate_response(model, tokenizer, input_text, max_length=128):
    input_ids = tokenizer.encode(input_text, return_tensors='tf')
    attention_mask = tf.ones_like(input_ids)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Select a subset of validation data for evaluation
eval_samples = min(50, len(X_val))
subset_X = X_val[:eval_samples]
subset_y = y_val[:eval_samples]

# Calculate BLEU scores
smoother = SmoothingFunction()
bleu_scores = []
generated_responses = []
reference_responses = []
input_texts = []

print("\nGenerating responses for evaluation...")
for i in range(eval_samples):
    input_text = tokenizer.decode(subset_X[i], skip_special_tokens=True)
    reference = tokenizer.decode(subset_y[i], skip_special_tokens=True)
    generated = generate_response(model, tokenizer, input_text)
    
    input_texts.append(input_text)
    reference_responses.append(reference)
    generated_responses.append(generated)
    
    # Tokenize for BLEU calculation
    ref_tokens = [reference.split()]
    gen_tokens = generated.split()
    
    bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoother.method1)
    bleu_scores.append(bleu)

avg_bleu = np.mean(bleu_scores)
print(f"\nAverage BLEU Score: {avg_bleu:.4f}")

# Calculate perplexity
def calculate_perplexity(model, input_ids, labels):
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    perplexity = tf.exp(loss)
    return perplexity.numpy()

# Calculate perplexity on validation set
val_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': X_val, 'attention_mask': attn_val},
    y_val
)).batch(8)

perplexity_scores = []
for batch, (inputs, labels) in enumerate(val_dataset):
    perplexity = calculate_perplexity(model, inputs, labels)
    perplexity_scores.append(perplexity)
    if batch >= 10:  # Limit to 10 batches for speed
        break

avg_perplexity = np.mean(perplexity_scores)
print(f"Average Perplexity: {avg_perplexity:.4f}")

# Calculate response relevance (simple keyword matching)
def calculate_relevance(question, response, domain_keywords):
    question_lower = question.lower()
    response_lower = response.lower()
    
    # Extract keywords from question
    question_keywords = set([word for word in question_lower.split() if word not in stop_words])
    
    # Count domain keywords in response
    domain_keyword_count = sum(1 for keyword in domain_keywords if keyword in response_lower)
    
    # Count question keywords in response
    question_keyword_count = sum(1 for keyword in question_keywords if keyword in response_lower)
    
    # Calculate relevance score (0-1)
    if len(question_keywords) > 0:
        relevance = min(1.0, (domain_keyword_count * 0.5 + question_keyword_count) / (len(question_keywords) + 1))
    else:
        relevance = min(1.0, domain_keyword_count * 0.5)
    
    return relevance

# Domain keywords for agriculture
agriculture_keywords = [
    'crop', 'farm', 'plant', 'harvest', 'soil', 'seed', 'water', 
    'fertilizer', 'pest', 'disease', 'agriculture', 'grow', 'field',
    'vegetable', 'fruit', 'irrigation', 'weather', 'season', 'rain',
    'drought', 'yield', 'organic', 'compost', 'manure', 'livestock'
]

# Calculate relevance scores
relevance_scores = []
for i in range(len(input_texts)):
    relevance = calculate_relevance(input_texts[i], generated_responses[i], agriculture_keywords)
    relevance_scores.append(relevance)

avg_relevance = np.mean(relevance_scores)
print(f"Average Response Relevance: {avg_relevance:.4f}")

# Create evaluation results dataframe
results_df = pd.DataFrame({
    'Question': input_texts,
    'Reference': reference_responses,
    'Generated': generated_responses,
    'BLEU': bleu_scores,
    'Relevance': relevance_scores
})

# Save evaluation results
results_df.to_csv('evaluation_results.csv', index=False)
print("Evaluation results saved to 'evaluation_results.csv'")

# Visualize results
plt.figure(figsize=(12, 6))

# BLEU score distribution
plt.subplot(1, 2, 1)
sns.histplot(bleu_scores, kde=True)
plt.title(f'BLEU Score Distribution (Avg: {avg_bleu:.4f})')
plt.xlabel('BLEU Score')
plt.ylabel('Count')

# Relevance score distribution
plt.subplot(1, 2, 2)
sns.histplot(relevance_scores, kde=True)
plt.title(f'Relevance Score Distribution (Avg: {avg_relevance:.4f})')
plt.xlabel('Relevance Score')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('evaluation_metrics.png')
print("Evaluation visualizations saved to 'evaluation_metrics.png'")

# Print qualitative evaluation examples
print("\nQualitative Evaluation Examples:")
for i in range(min(5, len(input_texts))):
    print(f"\nExample {i + 1}:")
    print(f"Input: {input_texts[i]}")
    print(f"Reference: {reference_responses[i]}")
    print(f"Generated: {generated_responses[i]}")
    print(f"BLEU Score: {bleu_scores[i]:.4f}")
    print(f"Relevance Score: {relevance_scores[i]:.4f}")
    print("---")