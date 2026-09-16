import re
import html
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

st.set_page_config(page_title="SEC Filing Research Assistant", layout="wide")

# Styling to match Primo Research Assistant layout
st.markdown("""
<style>
    .main-container {
        max-width: 980px;
        margin: 0 auto;
    }
    .query-box {
        background-color: #2b579a;
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 600;
        padding: 14px 20px;
        border-radius: 8px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .sources-header {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-bottom: 28px;
    }
    .source-mini-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        height: 125px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        font-size: 0.8rem;
    }
    .source-tag {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #2563eb;
    }
    .source-title {
        font-weight: 600;
        color: #1e293b;
        line-height: 1.3;
        margin-top: 4px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .source-meta {
        color: #64748b;
        font-size: 0.72rem;
    }
    .overview-section {
        background-color: #ffffff;
        border-top: 1px solid #e2e8f0;
        padding-top: 18px;
    }
    .overview-header {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .overview-body {
        color: #334155;
        font-size: 0.93rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_retriever():
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = None
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    return index.as_retriever(similarity_top_k=4)

with st.spinner("Initializing index..."):
    retriever = init_retriever()

def clean_sec_text(raw_text: str) -> str:
    text = html.unescape(raw_text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

st.title("Research Assistant")
query = st.text_input("Ask a research question regarding SEC corporate disclosures:", placeholder="e.g. liquidity risks, litigation, or intellectual property")

if st.button("Search", type="primary") or query:
    if query.strip():
        with st.spinner("Searching filing disclosures..."):
            nodes = retriever.retrieve(query)
            
            # Query Header Bar
            st.markdown(f"""
            <div class="query-box">
                <span>💬</span>
                <span>{query}</span>
            </div>
            """, unsafe_allow_html=True)
            
            if not nodes:
                st.info("No matching disclosure records found.")
            else:
                # Top horizontal cards
                st.markdown('<div class="sources-header">📁 Sources</div>', unsafe_allow_html=True)
                cols = st.columns(len(nodes))
                
                cleaned_chunks = []
                for idx, (col, node) in enumerate(zip(cols, nodes), start=1):
                    cleaned = clean_sec_text(node.text)
                    cleaned_chunks.append((idx, node.metadata.get("file_name", f"Filing {idx}"), cleaned))
                    
                    with col:
                        st.markdown(f"""
                        <div class="source-mini-card">
                            <div>
                                <span class="source-tag">📄 Source {idx}</span>
                                <div class="source-title">{node.metadata.get('file_name', 'SEC Form 10-K')}</div>
                            </div>
                            <div class="source-meta">{cleaned[:65]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Synthesis / Overview section below
                st.markdown("""
                <div class="overview-section">
                    <div class="overview-header">✨ Overview of disclosures</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Synthesized combined excerpts
                for idx, filename, excerpt in cleaned_chunks:
                    with st.expander(f"Source {idx} Excerpt ({filename})", expanded=(idx == 1)):
                        st.write(excerpt)
    else:
        st.warning("Please enter a research topic to search.")
