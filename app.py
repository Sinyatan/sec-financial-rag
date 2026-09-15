import os

# Constrain thread thrashing on limited CPU cores
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "1"

import gradio as gr
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# Matching lightweight FastEmbed model
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

# Retrieve top 2 matches to minimize similarity scan time
retriever = index.as_retriever(similarity_top_k=2)

def answer_query(message, history):
    if not message or not message.strip():
        return "Please enter a question."
    nodes = retriever.retrieve(message)
    if not nodes:
        return "No relevant excerpts found."
    results = []
    for i, node in enumerate(nodes, start=1):
        source = node.metadata.get("file_name", "SEC Document")
        results.append(f"**Result {i} (Source: {source})**\n\n{node.text.strip()}")
    return "\n\n---\n\n".join(results)

demo = gr.ChatInterface(
    fn=answer_query,
    title="SEC Financial Filing RAG",
    description="Query indexed SEC filings."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)