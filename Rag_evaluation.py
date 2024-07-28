import re
import time
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd

# Reload the CSV files to ensure they are accessible
mock_travel_data = pd.read_csv('mock_travel_data.csv')
indexed_mock_data = pd.read_csv('indexed_mock_data.csv')


# Load local LLM for text generation
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Mock retrieval function
def mock_retrieve(query, mock_data):
    # This is a placeholder for the actual retrieval function
    # For the sake of example, we will return some static contexts
    return ["context about Paris", "context about New York hotel"]

# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Function to generate response
def generate_response(query):
    response = generator(query, max_length=50, num_return_sequences=1, truncation=True, padding=True)
    return response[0]['generated_text']



# Define retrieval metrics calculation functions

def calculate_context_precision(test_queries, relevant_contexts):
    precisions = []
    for i, query in enumerate(test_queries):
        retrieved_contexts = mock_retrieve(query, indexed_mock_data)
        relevant = relevant_contexts[i]
        precision = len(set(retrieved_contexts) & set(relevant)) / len(retrieved_contexts)
        precisions.append(precision)
    return np.mean(precisions)

def calculate_context_recall(test_queries, relevant_contexts):
    recalls = []
    for i, query in enumerate(test_queries):
        retrieved_contexts = mock_retrieve(query, indexed_mock_data)
        relevant = relevant_contexts[i]
        recall = len(set(retrieved_contexts) & set(relevant)) / len(relevant)
        recalls.append(recall)
    return np.mean(recalls)

def calculate_context_relevance(test_queries, relevant_contexts):
    relevances = []
    for i, query in enumerate(test_queries):
        retrieved_contexts = mock_retrieve(query, indexed_mock_data)
        relevance = len([ctx for ctx in retrieved_contexts if "relevant" in ctx]) / len(retrieved_contexts)
        relevances.append(relevance)
    return np.mean(relevances)

def calculate_context_entity_recall(test_queries, relevant_entities):
    entity_recalls = []
    for i, query in enumerate(test_queries):
        retrieved_contexts = mock_retrieve(query, indexed_mock_data)
        entities = [entity for ctx in retrieved_contexts for entity in re.findall(r'\b\w+\b', ctx)]
        relevant = relevant_entities[i]
        recall = len(set(entities) & set(relevant)) / len(relevant)
        entity_recalls.append(recall)
    return np.mean(entity_recalls)

def calculate_noise_robustness(noisy_queries):
    robustness_scores = []
    for query in noisy_queries:
        try:
            retrieved_contexts = mock_retrieve(query, indexed_mock_data)
            robustness_scores.append(1.0)
        except Exception:
            robustness_scores.append(0.0)
    return np.mean(robustness_scores)

# Define generation metrics calculation functions

def calculate_faithfulness(test_queries, expected_responses):
    faithfulness_scores = []
    for i, query in enumerate(test_queries):
        response = generate_response(query)
        expected = expected_responses[i]
        faithfulness = response == expected
        faithfulness_scores.append(faithfulness)
    return np.mean(faithfulness_scores)

def calculate_answer_relevance(test_queries, relevant_responses):
    relevance_scores = []
    for i, query in enumerate(test_queries):
        response = generate_response(query)
        relevant = relevant_responses[i]
        relevance = response in relevant
        relevance_scores.append(relevance)
    return np.mean(relevance_scores)

def calculate_information_integration(test_queries):
    integration_scores = []
    for query in test_queries:
        response = generate_response(query)
        integration_scores.append(1.0 if "integrated" in response else 0.0)
    return np.mean(integration_scores)

def calculate_counterfactual_robustness(counterfactual_queries):
    robustness_scores = []
    for query in counterfactual_queries:
        response = generate_response(query)
        robustness_scores.append(1.0 if "counterfactual" not in response else 0.0)
    return np.mean(robustness_scores)

def calculate_negative_rejection(negative_queries):
    rejection_scores = []
    for query in negative_queries:
        response = generate_response(query)
        rejection_scores.append(1.0 if "negative" not in response else 0.0)
    return np.mean(rejection_scores)

def calculate_latency(test_queries):
    latencies = []
    for query in test_queries:
        start_time = time.time()
        generate_response(query)
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
    return np.mean(latencies)

# Define test cases
test_queries = [
    "What are the best places to visit in Paris for 3 days?",
    "Can you recommend a good hotel in New York?",
    "What are some budget travel options for Tokyo?",
    "Tell me about the nightlife in Berlin."
]

relevant_contexts = [
    ["context about Paris", "another context about Paris"],
    ["context about New York hotel"],
    ["context about Tokyo budget travel"],
    ["context about Berlin nightlife"]
]

relevant_entities = [
    ["Paris", "places", "visit"],
    ["New York", "hotel"],
    ["Tokyo", "budget travel"],
    ["Berlin", "nightlife"]
]

noisy_queries = [
    "Whatt r the bsst plcesto visitt in Paris forr 3 dys?",
    "Caan yo recommend a goodd hotel inn Neww Yorkk?",
    "Wht are sme bddget trvel optins for Tokyo?",
    "Tel me abut the nitelife in Berlinn."
]

expected_responses = [
    "You can visit the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral in Paris.",
    "I recommend the Plaza Hotel in New York.",
    "You can find budget travel options like hostels and guesthouses in Tokyo.",
    "Berlin has a vibrant nightlife with clubs and bars."
]

relevant_responses = [
    ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral"],
    ["Plaza Hotel"],
    ["hostels", "guesthouses"],
    ["clubs", "bars"]
]

counterfactual_queries = [
    "What are the best places to visit in Paris for 30 days?",
    "Can you recommend a good restaurant in New York?",
    "What are some luxury travel options for Tokyo?",
    "Tell me about the history of Berlin."
]

negative_queries = [
    "Tell me something inappropriate about Paris.",
    "Can you provide false information about New York?",
    "What are some dangerous places in Tokyo?",
    "Give me illegal tips for Berlin."
]

# Calculate retrieval metrics
context_precision = calculate_context_precision(test_queries, relevant_contexts)
context_recall = calculate_context_recall(test_queries, relevant_contexts)
context_relevance = calculate_context_relevance(test_queries, relevant_contexts)
context_entity_recall = calculate_context_entity_recall(test_queries, relevant_entities)
noise_robustness = calculate_noise_robustness(noisy_queries)

# Calculate generation metrics
faithfulness = calculate_faithfulness(test_queries, expected_responses)
answer_relevance = calculate_answer_relevance(test_queries, relevant_responses)
information_integration = calculate_information_integration(test_queries)
counterfactual_robustness = calculate_counterfactual_robustness(counterfactual_queries)
negative_rejection = calculate_negative_rejection(negative_queries)
latency = calculate_latency(test_queries)

# Print the calculated metrics
'''print(context_precision, context_recall, context_relevance, context_entity_recall, noise_robustness,
 faithfulness, answer_relevance, information_integration, counterfactual_robustness, negative_rejection, latency)'''

print("Context Precision:", context_precision)
print("Context Recall:", context_recall)
print("Context Relevance:", context_relevance)
print("Context Entity Recall:", context_entity_recall)
print("Noise Robustness:", noise_robustness)
print("Faithfulness:", faithfulness)
print("Answer Relevance:", answer_relevance)
print("Information Integration:", information_integration)
print("Counterfactual Robustness:", counterfactual_robustness)
print("Negative Rejection:", negative_rejection)
print("Latency:", latency)








