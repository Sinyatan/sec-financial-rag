import re
import html
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

st.set_page_config(page_title="SEC Financial Filing RAG Assistant", layout="wide")

# Modern card styling matching enterprise portals
st.markdown("""
<style>
    .result-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .badge-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 10px;
    }
    .source-badge {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #1e3a8a;
        background-color: #dbeafe;
        padding: 3px 10px;
        border-radius: 4px;
        display: inline-block;
    }
    .filing-badge {
        font-size: 0.75rem;
        font-weight: 500;
        color: #475569;
        background-color: #f1f5f9;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .excerpt-text {
        color: #334155;
        font-size: 0.93rem;
        line-height: 1.6;
        margin: 0;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 SEC Financial Filing RAG Assistant")
st.caption("Query 10-K and 10-Q corporate disclosures with grounded semantic search.")

@st.cache_resource
def init_retriever():
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = None
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    return index.as_retriever(similarity_top_k=3)

with st.spinner("Initializing filing index..."):
    retriever = init_retriever()

query = st.text_input("Enter your research question (e.g. liquidity risks, credit exposure, litigation):")

def clean_sec_text(raw_text: str) -> str:
    text = html.unescape(raw_text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

if st.button("Search Filings", type="primary") or query:
    if query.strip():
        with st.spinner("Retrieving relevant disclosure excerpts..."):
            nodes = retriever.retrieve(query)
            if not nodes:
                st.info("No matching sections found.")
            for idx, node in enumerate(nodes, start=1):
                source_file = node.metadata.get("file_name", "SEC Form 10-K")
                cleaned_text = clean_sec_text(node.text)
                
                st.markdown(f"""
                <div class="result-card">
                    <div class="badge-bar">
                        <span class="source-badge">Source {idx}</span>
                        <span class="filing-badge">📄 {source_file}</span>
                    </div>
                    <div class="excerpt-text">{cleaned_text}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a question to search.")
