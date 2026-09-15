import os
import gradio as gr
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Matching local embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

# Use direct retriever mode to pull matching document sections cleanly
retriever = index.as_retriever(similarity_top_k=3)

def answer_query(message, history):
    if not message or not message.strip():
        return "Please enter a question."
    
    nodes = retriever.retrieve(message)
    if not nodes:
        return "No relevant information found in the SEC filings."
    
    results = []
    for i, node in enumerate(nodes, start=1):
        source = node.metadata.get("file_name", "SEC Document")
        score = f"{node.score:.3f}" if node.score is not None else "N/A"
        results.append(f"### Result {i} (Source: {source} | Match: {score})\n{node.text.strip()}\n")
        
    return "\n---\n\n".join(results)

demo = gr.ChatInterface(
    fn=answer_query,
    title="SEC Financial Filing RAG",
    description="Search and extract verified excerpts from indexed SEC filings."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))