DocuRetrieve

## Overview

DocuRetrieve is a Retrieval-Augmented Generation (RAG) based customer support assistant designed to answer questions about Zepto policies.

It uses a policy document corpus, Sentence Transformers (all-MiniLM-L6-v2) for embeddings, ChromaDB for vector storage and retrieval, Pydantic for structured responses, and FastAPI for API access.

The default mode uses `MOCK_LLM`, so the graded version works without an external LLM API.

## Installation
1) clone the repository:

git clone https://github.com/charan33sai33/DocuRetrieve-RAG-Project.git

2) Move into the project directory:

cd DocuRetrieve-RAG-Project

3) Create a virtual environment:

python -m venv venv

4) From the project root Install dependencies:

pip install -r requirements.txt

## How to Run

python -m uvicorn support_assistant:app --reload


open fastapi documentation at http://127.0.0.1:8000/docs
POST /ask Application provides the following endpoint:

## Example 1 - Policy Question

Request:
{
  "query": "What is the delivery fee?"
}
Response:
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_02"
  ],
  "confidence": 1.0
}
## Example 2 - General Question

Request:
{
  "query": "What is the capital of India?"
}

Response:

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
            

## RAG Architecture

Policy documents
      |
      v
Read the documents
      |
      v
Create embeddings using all-MiniLM-L6-v2
      |
      v
Store embeddings in ChromaDB
      |
      v
User asks a question
      |
      v
classify_intent
      |
      +----------------------+
      |                      |
      v                      v
policy_question       general_question
      |                      |
      v                      v
retrieve_and_answer   direct_answer
      |
      v
Pydantic response
      |
      v
FastAPI /ask


## MOCK_LLM

The default mode is mock mode because MOCK_LLM is unset.
Intent classification uses the required keyword heuristic.
ChromaDB retrieval runs normally for policy questions.
No external LLM API is called.
Policy answers are generated from the top retrieved chunk.
General questions return a fixed response.
The optional real-LLM path can be enabled by setting:

## Pydantic Response

The API response follows this structure:
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 1.0
}

The sources field contains the document IDs used for a policy answer and is empty for a general question.
The confidence value is between 0 and 1.
## Docker

Build the Docker image from the project root:  docker build -t zepto-support .

Run the Docker container: docker run --rm -p 7860:7860 zepto-support

Open: http://127.0.0.1:7860/docs