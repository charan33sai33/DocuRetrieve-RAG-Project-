import os
from typing import TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END


folder = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.join(folder, "docs")
db_path = os.path.join(folder, "chroma_db")

mock_llm = os.getenv("MOCK_LLM", "1") != "0"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=db_path)

collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)

doc_ids = []
documents = []

for i in range(1, 9):
    doc_id = f"doc_{i:02d}"
    file_path = os.path.join(docs_path, f"{doc_id}.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing file: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().strip()

    doc_ids.append(doc_id)
    documents.append(text)

embeddings = embedder.encode(
    documents
).tolist()

collection.upsert(
    ids=doc_ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=[
        {"document": doc_id}
        for doc_id in doc_ids
    ]
)


class State(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)


prompt = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the information in the retrieved Zepto policy documents.

TASK:
Answer the user's question clearly and directly.

FORMAT:
Return JSON with answer, sources and confidence.

LENGTH:
Keep the answer short and relevant.

NEGATIVE CONSTRAINT:
Do not use information that is not present in the provided context.

FEW-SHOT EXAMPLE:
Question: What is the delivery fee below INR 149?
Context: Orders below INR 149 incur a flat INR 25 delivery fee.
Answer: Orders below INR 149 incur a flat INR 25 delivery fee.
"""


keywords = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


def call_llm(text):
    from groq import Groq

    groq = Groq(
        api_key=os.environ["GROQ_API_KEY"]
    )

    return groq.chat.completions.create(
        model=os.getenv(
            "GROQ_MODEL",
            "llama-3.1-8b-instant"
        ),
        messages=[
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0
    ).choices[0].message.content


def get_llm_answer(text):
    for _ in range(3):
        try:
            result = call_llm(text)
            return AskResponse.model_validate_json(result)
        except Exception:
            text += """
Return only valid JSON:
{
    "answer": "string",
    "sources": ["string"],
    "confidence": 1.0
}
"""

    return AskResponse(
        answer="Unable to generate a valid response.",
        sources=[],
        confidence=0.0
    )


def classify_intent(state):
    query = state["query"].lower()

    if mock_llm:
        for word in keywords:
            if word in query:
                return {
                    "intent": "policy_question"
                }

        return {
            "intent": "general_question"
        }

    result = call_llm(
        f"""
Classify the query as exactly one of:
policy_question
general_question

Query:
{state["query"]}

Return only one label.
"""
    )

    if "policy_question" in result.lower():
        return {
            "intent": "policy_question"
        }

    return {
        "intent": "general_question"
    }


def retrieve_and_answer(state):
    query_embedding = embedder.encode(
        [state["query"]]
    ).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    source_ids = result["ids"][0]
    source_texts = result["documents"][0]

    if mock_llm:
        return {
            "answer": (
                "Based on the retrieved context: "
                + source_texts[0][:200]
            ),
            "sources": source_ids,
            "confidence": 1.0
        }

    context = "\n\n".join(source_texts)

    llm_prompt = f"""
{prompt}

Context:
{context}

Question:
{state["query"]}

Return only JSON.
"""

    result = get_llm_answer(llm_prompt)

    return {
        "answer": result.answer,
        "sources": source_ids,
        "confidence": result.confidence
    }


def direct_answer(state):
    if mock_llm:
        return {
            "answer": "I can only answer questions about Zepto policies right now.",
            "sources": [],
            "confidence": 1.0
        }

    llm_prompt = f"""
{prompt}

There is no policy context for this question.

Question:
{state["query"]}

Return only JSON.
"""

    result = get_llm_answer(llm_prompt)

    return {
        "answer": result.answer,
        "sources": [],
        "confidence": result.confidence
    }


def choose_path(state):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


graph = StateGraph(State)

graph.add_node(
    "classify_intent",
    classify_intent
)

graph.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph.add_node(
    "direct_answer",
    direct_answer
)

graph.add_edge(
    START,
    "classify_intent"
)

graph.add_conditional_edges(
    "classify_intent",
    choose_path,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph.add_edge(
    "retrieve_and_answer",
    END
)

graph.add_edge(
    "direct_answer",
    END
)

app_graph = graph.compile()


def ask_question(query):
    result = app_graph.invoke(
        {
            "query": query,
            "intent": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0
        }
    )

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )


app = FastAPI()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return ask_question(request.query)


if __name__ == "__main__":
    print("Documents in ChromaDB:", collection.count())

    result1 = ask_question(
        "What is the delivery fee?"
    )

    result2 = ask_question(
        "What is the capital of India?"
    )

    print("\nPolicy question:")
    print(result1.model_dump_json())

    print("\nGeneral question:")
    print(result2.model_dump_json())