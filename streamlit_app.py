import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

st.set_page_config(page_title="SEC Financial Filing RAG Assistant", layout="wide")

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

if st.button("Search Filings", type="primary") or query:
    if query.strip():
        with st.spinner("Retrieving relevant disclosure excerpts..."):
            nodes = retriever.retrieve(query)
            if not nodes:
                st.info("No matching sections found.")
            for idx, node in enumerate(nodes, start=1):
                source_file = node.metadata.get("file_name", "SEC Filing")
                with st.expander(f"Result {idx} — Source: {source_file}", expanded=True):
                    st.markdown(node.text.strip())
    else:
        st.warning("Please enter a question to search.")