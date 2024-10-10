import os
import re
import nltk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. Text Preprocessing

# Download stopwords from nltk
stop_words = set(stopwords.words('english'))

# File paths
train_dir = r'C:Data/train'
test_dir = r'C:Data/test'

# Clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', re.sub(r'\d+', '', text.lower()))
    return ' '.join([word for word in text.split() if word not in stop_words])

# Load and preprocess dataset
def load_data(directory):
    reviews, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(directory, label_type)
        for fname in filter(lambda f: f.endswith('.txt'), os.listdir(dir_name)):
            with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                reviews.append(clean_text(f.read()))
                labels.append(1 if label_type == 'pos' else 0)
    return reviews, labels

# Load training and test data
train_reviews, train_labels = load_data(train_dir)
test_reviews, test_labels = load_data(test_dir)

# 2. Tokenization and Padding
vocab_size = 10000
max_length = 200
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_reviews)

train_padded = pad_sequences(tokenizer.texts_to_sequences(train_reviews), maxlen=max_length, truncating='post')
test_padded = pad_sequences(tokenizer.texts_to_sequences(test_reviews), maxlen=max_length, truncating='post')

train_labels, test_labels = np.array(train_labels), np.array(test_labels)

# 3. Model Definition and Training
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)), # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(train_padded, train_labels, epochs=10, verbose=2)

# 4. Final Evaluation on the Test Set and Confusion Matrix
test_loss, test_acc = model.evaluate(test_padded, test_labels, verbose=2)
test_preds = (model.predict(test_padded) > 0.5).astype(int)
conf_matrix = confusion_matrix(test_labels, test_preds)

# Print final test accuracy and confusion matrix
print(f"Test Accuracy: {test_acc:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# 5. Plot Loss and Accuracy Graph
def plot_metrics(history):
    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


plot_metrics(history)

