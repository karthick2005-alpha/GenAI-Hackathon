import tensorflow as tf
import numpy as np
import os
import time

# Load dataset (you can use any text dataset; here we use Shakespeare's work for simplicity)
path_to_file = 'shakespeare.txt'  # Path to your text file

# Ensure the file exists
if not os.path.exists(path_to_file):
    print(f"File not found: {path_to_file}")
else:
    with open(path_to_file, 'r') as f:
        text = f.read()

    # Create a mapping of unique characters to indices and vice versa
    vocab = sorted(set(text))  # All unique characters in the text
    char_to_index = {char: index for index, char in enumerate(vocab)}
    index_to_char = {index: char for index, char in enumerate(vocab)}

    # Convert text into integers
    text_as_int = np.array([char_to_index[c] for c in text])

    # Define the length of input sequences and the batch size
    seq_length = 100
    batch_size = 64

    # Create training sequences and labels
    def create_sequences(text_as_int, seq_length):
        input_seq = []
        output_seq = []
        for i in range(0, len(text_as_int) - seq_length, 1):
            input_seq.append(text_as_int[i:i + seq_length])
            output_seq.append(text_as_int[i + seq_length])
        return np.array(input_seq), np.array(output_seq)

    X, y = create_sequences(text_as_int, seq_length)

    # Reshape X for the model
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (num_samples, seq_length, 1)
    X = X / float(len(vocab))  # Normalize input

    # Define the LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dense(len(vocab), activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    # Train the model
    model.fit(X, y, batch_size=batch_size, epochs=50)

    # Generate text from the trained model
    def generate_text(model, start_string, char_to_index, index_to_char, num_generate=500):
        input_eval = [char_to_index[s] for s in start_string]
        input_eval = np.expand_dims(input_eval, 0)

        predicted_text = start_string
        model.reset_states()

        for _ in range(num_generate):
            predictions = model(input_eval)
            predictions = predictions[:, -1, :]  # Get the last timestep prediction
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            predicted_char = index_to_char[predicted_id]
            predicted_text += predicted_char

            input_eval = np.expand_dims([predicted_id], 0)

        return predicted_text

    start_string = "ROMEO: "
    generated_text = generate_text(model, start_string, char_to_index, index_to_char)
    print(generated_text)
    # Install necessary libraries
!pip install transformers torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'  # You can choose other variants like 'gpt2-medium', 'gpt2-large', etc.
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Encode input text to tokens
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

# Decode and print the output text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
