# Travel_Recommendation_chatbot
# Travel Itinerary Recommendation Chatbot

This project implements a Travel Itinerary Recommendation Chatbot using Retrieval-Augmented Generation (RAG) and FAISS for vector similarity search. The chatbot leverages a Large Language Model (LLM) to understand and process natural language queries and provides personalized travel recommendations based on pre-existing data.

## Features

- **Personalized Recommendations**: The chatbot generates travel itineraries based on user preferences and needs.
- **Retrieval-Augmented Generation**: Combines the strengths of LLM for natural language understanding with the efficiency of a vector database for data retrieval.
- **Streamlit Interface**: An easy-to-use web interface for interacting with the chatbot.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Install the required libraries

### Installation

1. **Clone the repository**:
    ```bash
    git clone (https://github.com/Nishi-Gandhi/Travel_itenary_chatbot/)
    cd travel-itinerary-chatbot
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Generate mock data**:
    The project includes a script to generate mock travel data using an open-source LLM.
    ```bash
    python generate_mock_data.py
    ```

### Running the Streamlit App

1. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2. **Open the app in your browser**:
    Streamlit will automatically open a new tab in your default web browser. If it doesn't, you can manually open `http://localhost:8501`.

## Usage

1. **Enter your travel preferences**: In the text input box, type in your travel preferences or needs, such as "I want to visit a museum in the afternoon" or "I need a travel plan that includes a morning hike and an evening show."
2. **Get Recommendations**: Click the "Get Recommendations" button to receive personalized travel itineraries based on your input.
3. **View Results**: The chatbot will display the question, answer, and the context used to generate the response.

## Project Structure

- **app.py**: Main file for running the Streamlit application.
- **generate_mock_data.py**: Script for generating mock travel data using GPT-2.
- **requirements.txt**: List of dependencies required for the project.
- **travel_data.csv**: Generated mock travel data (if already provided, otherwise created by `generate_mock_data.py`).

## Acknowledgements

This project uses the following libraries and models:
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://github.com/streamlit/streamlit)

## License

This project is licensed under the MIT License.

