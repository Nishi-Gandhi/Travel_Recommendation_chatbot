import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate text
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Prompts for generating travel data
prompts = [
    "Describe a perfect day in Paris, including places to visit, restaurants to eat at, and activities to do.",
    "Provide a detailed itinerary for a 3-day trip to New York City.",
    "What are the best places to visit in Tokyo for someone who loves technology and anime?",
    "List some family-friendly activities to do in London during the summer.",
    "Create a travel guide for a food lover visiting Bangkok."
]

# Generate data
data = []
for prompt in prompts:
    for _ in range(5):  # Generate 5 variations for each prompt
        generated_text = generate_text(prompt, max_length=150)
        data.append({
            "Prompt": prompt,
            "Generated Text": generated_text
        })

# Create a DataFrame and save it as CSV
df = pd.DataFrame(data)
df.to_csv("travel_data.csv", index=False)

print("Mock data generated and saved to travel_data.csv")
