
import os
import random
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

folder = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(folder, "chroma_db")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=db_path)

collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)

questions = [
    {"question": "Can I combine a gift voucher with another way of paying?",
     "expected_doc": "doc_07"},

    {"question": "What should I do if my delivery is running late and the rider has not moved?",
     "expected_doc": "doc_04"},

    {"question": "Can I return a personal care item after opening its packaging?",
     "expected_doc": "doc_02"},

    {"question": "At what point does cancelling an order stop being free?",
     "expected_doc": "doc_05"},

    {"question": "What advantages do customers receive with Zepto Pass?",
     "expected_doc": "doc_03"},

    {"question": "How should I report groceries that arrived damaged?",
     "expected_doc": "doc_06"},

    {"question": "What is the usual delivery time for an order?",
     "expected_doc": "doc_01"},

    {"question": "Can I contact customer support late at night?",
     "expected_doc": "doc_08"},

    {"question": "How long can I use a Zepto gift card after buying it?",
     "expected_doc": "doc_07"},

    {"question": "Will I get my membership fee back if I cancel my pass?",
     "expected_doc": "doc_03"},

    {"question": "Where can I follow my delivery person's live location?",
     "expected_doc": "doc_04"},

    {"question": "How long do I have to report spoiled items?",
     "expected_doc": "doc_02"},

    {"question": "What happens if no delivery rider is available for a packed order?",
     "expected_doc": "doc_05"},

    {"question": "Does Zepto offer phone support or email assistance?",
     "expected_doc": "doc_08"},

    {"question": "What is the minimum cart value for free delivery?",
     "expected_doc": "doc_01"},

    {"question": "What proof should I provide for an expensive missing-item complaint?",
     "expected_doc": "doc_06"},

    {"question": "How much does the premium Zepto membership cost each month?",
     "expected_doc": "doc_03"},

    {"question": "Can I cancel an order in the app once packing has started?",
     "expected_doc": "doc_05"},

    {"question": "How quickly should a refund appear on my original payment method?",
     "expected_doc": "doc_02"},

    {"question": "Are gift cards available in different monetary values?",
     "expected_doc": "doc_07"},

    {"question": "When can I start tracking the rider assigned to my order?",
     "expected_doc": "doc_04"},

    {"question": "What should I do when an item is missing from my delivered order?",
     "expected_doc": "doc_06"},

    {"question": "Is there an additional charge for faster delivery?",
     "expected_doc": "doc_01"},

    {"question": "How soon does the email support team respond?",
     "expected_doc": "doc_08"},

    {"question": "Are unopened non-perishable products eligible for returns?",
     "expected_doc": "doc_02"}
]

# Shuffle questions so documents are tested in mixed order.
random.seed(42)
random.shuffle(questions)

stored_ids = set(collection.get()["ids"])
results = []

for item in questions:
    expected = item["expected_doc"]

    if expected not in stored_ids:
        print("Expected document missing:", expected)
        continue

    embedding = embedder.encode(
        [item["question"]]
    ).tolist()

    response = collection.query(
        query_embeddings=embedding,
        n_results=min(3, collection.count())
    )

    retrieved = response["ids"][0]
    hit = expected in retrieved

    results.append({
        "question": item["question"],
        "expected_document": expected,
        "retrieved_documents": ", ".join(retrieved),
        "top3_hit": hit,
        "top1_hit": retrieved[0] == expected
    })

df = pd.DataFrame(results)

if not df.empty:
    total = len(df)
    top3_correct = int(df["top3_hit"].sum())
    top1_correct = int(df["top1_hit"].sum())

    print("\nQuestions evaluated:", total)
    print(
        f"Top-1 accuracy: {top1_correct}/{total} "
        f"({top1_correct / total * 100:.2f}%)"
    )
    print(
        f"Top-3 accuracy: {top3_correct}/{total} "
        f"({top3_correct / total * 100:.2f}%)"
    )

    output_path = os.path.join(
        folder, "retrieval_results.csv"
    )
    df.to_csv(output_path, index=False)

    print("\nResults saved to:", output_path)
    print("\nDetailed results:")
    print(df.to_string(index=False))
else:
    print("No questions were evaluated.")
