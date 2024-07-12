# from transformers import AutoTokenizer, AutoModelForCausalLM
# from langchain.document_loaders import CSVLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch

# class AIAgent:
#     def __init__(self, model_name="gpt2", max_length=256):
#         self.max_length = max_length
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
#     def create_prompt(self, query, context):
#         prompt = f"""
#         You are an assistant for question-answering tasks for Retrieval Augmented Generation system. 
#         Use the following pieces of retrieved context to answer the question. 
#         If you don't know the answer, just say that you don't know. 
#         Use two sentences maximum and keep the answer concise.
#         Question: {query}
#         Context: {context}
#         Answer:
#         """
#         return prompt
    
#     def generate(self, query, retrieved_info):
#         prompt = self.create_prompt(query, retrieved_info)
#         input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
#         output = self.model.generate(input_ids, max_length=self.max_length)
#         answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
#         return prompt, answer

# class RAGSystem:
#     def __init__(self, ai_agent, num_retrieved_docs=3, model_name="all-MiniLM-L6-v2"):
#         self.num_docs = num_retrieved_docs
#         self.ai_agent = ai_agent
#         # Assuming data is already preprocessed and available in CSV format
#         loader = CSVLoader("travel_data.csv")
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
#         all_splits = text_splitter.split_documents(documents)
#         embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
#         # Create a FAISS index
#         self.vector_db = FAISS.from_documents(documents=all_splits, embedding=embeddings)
    
#     def retrieve(self, query):
#         docs = self.vector_db.similarity_search(query, k=self.num_docs)
#         return docs
    
#     def query(self, query):
#         context = self.retrieve(query)
#         data = " ".join([doc.page_content for doc in context[:self.num_docs]])
#         prompt, answer = self.ai_agent.generate(query, data[:500])
#         return {"question": query, "answer": answer, "context": data}

# app = Flask(__name__)
# CORS(app)
# ai_agent = AIAgent(model_name="gpt2")
# rag_system = RAGSystem(ai_agent)

# @app.route('/query', methods=['POST'])
# def query():
#     user_input = request.json.get("query")
#