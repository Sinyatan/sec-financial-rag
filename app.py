import os
import gradio as gr
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# 1. Models & Configuration (Render provides GOOGLE_API_KEY via environment variables)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
Settings.llm = GoogleGenAI(model="gemini-3-flash-preview")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Load Local Index
PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize",
)

# 3. Query Handler
def query_rag(user_question):
    if not user_question.strip():
        return "Please enter a question regarding the SEC filing."
    try:
        response = query_engine.query(user_question)
        return str(response)
    except Exception as e:
        return f"Error processing query: {str(e)}"

# 4. Gradio Interface
demo = gr.Interface(
    fn=query_rag,
    inputs=gr.Textbox(
        lines=2, 
        placeholder="Ask a question (e.g., What are the primary risk factors and revenue drivers?)...",
        label="SEC Document Query"
    ),
    outputs=gr.Textbox(lines=8, label="Analysis Response"),
    title="SEC Filings RAG Assistant",
    description="Financial analysis assistant powered by Google Gemini and LlamaIndex.",
    flagging_mode="never"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
