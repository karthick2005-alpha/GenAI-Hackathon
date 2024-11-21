# Install required libraries
# pip install transformers torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'  # You can also try 'gpt2-medium', 'gpt2-large', etc.
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Encode the prompt text
prompt_text = "best budget laptops"
inputs = tokenizer.encode(prompt_text, return_tensors='pt')

# Generate text based on the prompt
outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)