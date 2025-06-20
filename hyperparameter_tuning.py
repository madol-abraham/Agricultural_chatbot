import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("Loading dataset...")
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

print("Preprocessing data...")
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

print("Tokenizing data...")
# Tokenize questions and answers
questions_tokenized = tokenize_data(df['processed_question'])
answers_tokenized = tokenize_data(df['processed_answer'])

# Prepare input and target sequences
input_ids = questions_tokenized['input_ids']
attention_mask = questions_tokenized['attention_mask']
labels = answers_tokenized['input_ids']

print("Splitting data into train/validation sets...")
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

# Training function
@tf.function
def train_step(model, inputs, labels, optimizer, loss_fn, metric):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        logits = outputs.logits
        loss_value = loss_fn(labels, logits)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    metric(loss_value)

    return loss_value

# Validation function
@tf.function
def val_step(model, inputs, labels, loss_fn, metric):
    outputs = model(inputs, training=False)
    logits = outputs.logits
    loss_value = loss_fn(labels, logits)
    metric(loss_value)
    return loss_value

# Hyperparameter tuning setup
hyperparams = [
    {'learning_rate': 5e-5, 'batch_size': 8, 'epochs': 3},
    {'learning_rate': 3e-5, 'batch_size': 8, 'epochs': 3},
    {'learning_rate': 2e-5, 'batch_size': 8, 'epochs': 3},
    {'learning_rate': 5e-5, 'batch_size': 16, 'epochs': 3},
    {'learning_rate': 3e-5, 'batch_size': 16, 'epochs': 3},
    {'learning_rate': 2e-5, 'batch_size': 16, 'epochs': 3}
]

results = []

# Load base model for baseline performance
print("Loading base model for baseline performance measurement...")
base_model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Define loss function
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
metric = Mean(name='loss')

# Measure baseline performance
print("Measuring baseline performance...")
val_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': X_val, 'attention_mask': attn_val},
    y_val
)).batch(8)

metric.reset_state()
for val_batch, (val_inputs, val_labels) in enumerate(val_dataset):
    val_loss_value = val_step(base_model, val_inputs, val_labels, loss_fn, metric)

baseline_loss = metric.result().numpy()
baseline_perplexity = np.exp(baseline_loss)
print(f"Baseline Validation Loss: {baseline_loss:.4f}")
print(f"Baseline Perplexity: {baseline_perplexity:.4f}")

# Training loop with hyperparameter tuning
print("\nStarting hyperparameter tuning...")
for hp_idx, hp in enumerate(hyperparams):
    print(f"\nExperiment {hp_idx + 1}/{len(hyperparams)}")
    print(f"Training with hyperparameters: {hp}")
    
    # Reset model for each experiment
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    optimizer = Adam(learning_rate=hp['learning_rate'])
    
    # Recreate datasets with current batch size
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': X_train, 'attention_mask': attn_train},
        y_train
    )).shuffle(1000).batch(hp['batch_size'])
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': X_val, 'attention_mask': attn_val},
        y_val
    )).batch(hp['batch_size'])
    
    # Track metrics for each experiment
    train_losses = []
    val_losses = []
    
    # Training
    start_time = time.time()
    for epoch in range(hp['epochs']):
        print(f"\nEpoch {epoch + 1}/{hp['epochs']}")
        metric.reset_state()
        
        for batch, (inputs, labels) in enumerate(train_dataset):
            loss_value = train_step(model, inputs, labels, optimizer, loss_fn, metric)
            
            if batch % 10 == 0:
                print(f"Batch {batch}, Loss: {metric.result().numpy():.4f}")
        
        # Record training loss
        train_loss = metric.result().numpy()
        train_losses.append(train_loss)
        
        # Validation
        metric.reset_state()
        for val_batch, (val_inputs, val_labels) in enumerate(val_dataset):
            val_loss_value = val_step(model, val_inputs, val_labels, loss_fn, metric)
        
        val_loss = metric.result().numpy()
        val_losses.append(val_loss)
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Perplexity: {np.exp(val_loss):.4f}")
    
    training_time = time.time() - start_time
    
    # Calculate improvement over baseline
    final_val_loss = val_losses[-1]
    improvement = (baseline_loss - final_val_loss) / baseline_loss * 100
    
    # Store results
    results.append({
        'hyperparameters': hp,
        'final_train_loss': train_losses[-1],
        'final_val_loss': final_val_loss,
        'final_perplexity': np.exp(final_val_loss),
        'improvement': improvement,
        'training_time': training_time,
        'train_loss_history': train_losses,
        'val_loss_history': val_losses
    })
    
    # Save model if it's the best so far
    if hp_idx == 0 or final_val_loss < min(r['final_val_loss'] for r in results[:-1]):
        print(f"New best model found! Saving...")
        model.save_pretrained(f'fine_tuned_gpt2_chatbot')
        tokenizer.save_pretrained(f'fine_tuned_gpt2_chatbot')

# Display hyperparameter tuning results
print("\nHyperparameter Tuning Results:")
results_df = pd.DataFrame([
    {
        'Learning Rate': r['hyperparameters']['learning_rate'],
        'Batch Size': r['hyperparameters']['batch_size'],
        'Epochs': r['hyperparameters']['epochs'],
        'Train Loss': r['final_train_loss'],
        'Validation Loss': r['final_val_loss'],
        'Perplexity': r['final_perplexity'],
        'Improvement (%)': r['improvement'],
        'Training Time (s)': r['training_time']
    }
    for r in results
])

print(results_df)
results_df.to_csv('hyperparameter_tuning_results.csv', index=False)

# Find best model
best_idx = results_df['Validation Loss'].idxmin()
best_config = results_df.iloc[best_idx]
print(f"\nBest Configuration:")
print(f"Learning Rate: {best_config['Learning Rate']}")
print(f"Batch Size: {best_config['Batch Size']}")
print(f"Epochs: {best_config['Epochs']}")
print(f"Validation Loss: {best_config['Validation Loss']:.4f}")
print(f"Perplexity: {best_config['Perplexity']:.4f}")
print(f"Improvement over baseline: {best_config['Improvement (%)']:.2f}%")

# Visualize results
plt.figure(figsize=(15, 10))

# Plot validation loss for each configuration
plt.subplot(2, 2, 1)
for i, r in enumerate(results):
    hp = r['hyperparameters']
    label = f"lr={hp['learning_rate']}, bs={hp['batch_size']}"
    plt.plot(range(1, hp['epochs'] + 1), r['val_loss_history'], marker='o', label=label)
plt.axhline(y=baseline_loss, color='r', linestyle='--', label='Baseline')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss by Configuration')
plt.legend()
plt.grid(True)

# Plot training loss for each configuration
plt.subplot(2, 2, 2)
for i, r in enumerate(results):
    hp = r['hyperparameters']
    label = f"lr={hp['learning_rate']}, bs={hp['batch_size']}"
    plt.plot(range(1, hp['epochs'] + 1), r['train_loss_history'], marker='o', label=label)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss by Configuration')
plt.legend()
plt.grid(True)

# Plot final validation loss by learning rate
plt.subplot(2, 2, 3)
sns.barplot(x='Learning Rate', y='Validation Loss', hue='Batch Size', data=results_df)
plt.title('Validation Loss by Learning Rate and Batch Size')
plt.grid(True)

# Plot improvement percentage
plt.subplot(2, 2, 4)
sns.barplot(x='Learning Rate', y='Improvement (%)', hue='Batch Size', data=results_df)
plt.title('Improvement Over Baseline (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('hyperparameter_tuning_results.png')
print("Hyperparameter tuning visualizations saved to 'hyperparameter_tuning_results.png'")