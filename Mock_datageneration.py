import pandas as pd
import random
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample destinations, activities, and hotels
destinations = [
    "Paris", "London", "New York", "Tokyo", "Sydney", "Rome", "Dubai", "Barcelona",
    "Istanbul", "Amsterdam", "Hong Kong", "Bangkok", "Singapore", "Los Angeles", "San Francisco"
]
activities = [
    "art museums", "beaches", "nightlife", "historical sites", "shopping", "nature parks",
    "culinary tours", "sightseeing", "adventure sports", "cultural festivals"
]
hotels = [
    "Hilton", "Marriott", "Hyatt", "Sheraton", "Ritz-Carlton", "Four Seasons", "Holiday Inn", "Westin",
    "InterContinental", "Radisson", "Waldorf Astoria", "Mandarin Oriental", "St. Regis", "Peninsula", "Park Hyatt"
]
price_ranges = ["$1000-$2000", "$2000-$3000", "$3000-$4000", "$4000-$5000"]

# Function to generate a mock response
def generate_response(destination, activity1, activity2, hotel, price_range, query_type):
    if query_type == "package":
        response = f"Travel Package for {destination}:\n1. Day 1: Explore {activity1}.\n2. Day 2: Enjoy {activity2}.\n3. Day 3: Continue exploring the city.\n4. Day 4: Visit nearby attractions.\n5. Day 5: Departure.\nTotal Price: {price_range}"
    elif query_type == "budget":
        response = f"Budget Trip to {destination}:\n1. Day 1: Explore {activity1}.\n2. Day 2: Enjoy {activity2}.\n3. Day 3: Continue exploring the city.\n4. Day 4: Visit nearby attractions.\n5. Day 5: Departure.\nTotal Budget: {price_range}"
    elif query_type == "hotel":
        response = f"Suggested Hotel in {destination}: {hotel}.\nIt offers great amenities and is located close to major attractions."
    else:
        response = ""
    return response

# Generate mock data with validation
data = []
for i in range(1, 101):
    destination = random.choice(destinations)
    activity1, activity2 = random.sample(activities, 2)
    hotel = random.choice(hotels)
    price_range = random.choice(price_ranges)
    query_type = random.choice(["package", "budget", "hotel"])
    
    query = f"Tell me the travel package for {destination}" if query_type == "package" else f"Can you plan a budget trip to {destination} for {price_range}?" if query_type == "budget" else f"Suggest a hotel in {destination}"
    response = generate_response(destination, activity1, activity2, hotel, price_range, query_type)
    
    # Validate the response
    if not response:
        print(f"Error generating response for query: {query}")
        continue
    
    data.append({"id": i, "query": query, "response": response})

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('mock_travel_data.csv', index=False)

# Load mock data
mock_data = pd.read_csv('mock_travel_data.csv')

# Preprocess the data for embedding
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower().strip()
    return text

mock_data['cleaned_query'] = mock_data['query'].apply(preprocess_text)

# Generate embeddings for the queries
model = SentenceTransformer('all-MiniLM-L6-v2')

try:
    embeddings = model.encode(mock_data['cleaned_query'].tolist())
except Exception as e:
    print(f"Error generating embeddings: {e}")

# Ensure all embeddings are generated properly
if len(embeddings) != len(mock_data):
    print("Error: Mismatch between the number of embeddings and the number of queries")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save the index and data
index_path = 'faiss_mock_index'
data_path = 'indexed_mock_data.csv'
faiss.write_index(index, index_path)
mock_data.to_csv(data_path, index=False)
