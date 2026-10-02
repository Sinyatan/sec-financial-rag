import re
import streamlit as st
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

st.set_page_config(page_title="SEC Financial Filing RAG Assistant", layout="wide")

st.markdown("## 📊 SEC Financial Filing RAG Assistant")
st.caption("Query 10-K and 10-Q corporate disclosures with grounded semantic search.")


@st.cache_resource
def init_retriever():
  Settings.embed_model = HuggingFaceEmbedding(
      model_name="BAAI/bge-small-en-v1.5"
  )
  Settings.llm = None
  storage_context = StorageContext.from_defaults(persist_dir="./storage")
  index = load_index_from_storage(storage_context)
  return index.as_retriever(similarity_top_k=3)


def clean_html(raw_text):
  cleanr = re.compile("<.*?>")
  return re.sub(cleanr, "", raw_text)


with st.spinner("Initializing filing index..."):
  retriever = init_retriever()

query = st.text_input(
    "Enter your research question (e.g. liquidity risks, credit exposure,"
    " litigation):"
)

if st.button("Search Filings", type="primary") or query:
  if query.strip():
    with st.spinner("Retrieving relevant disclosure excerpts..."):
      nodes = retriever.retrieve(query)
      if not nodes:
        st.info("No matching sections found.")
      else:
        st.markdown("---")
        st.subheader("Sources")

        cols = st.columns(len(nodes))
        for idx, (col, node) in enumerate(zip(cols, nodes), start=1):
          source_file = node.metadata.get("file_name", f"Filing_{idx}")
          # Fallback to direct official SEC EDGAR search if URL is broken or missing
          sec_url = node.metadata.get("source_url")
          if not sec_url or "CIK=10-K" in sec_url:
            sec_url = (
                "https://www.sec.gov/edgar/searchedgar/companysearch"  # Safe default link
            )

          with col:
            card_html = f"""
                        <a href="{sec_url}" target="_blank" style="text-decoration: none; color: inherit;">
                            <div style="padding: 14px; border: 1px solid #d0d7de; border-radius: 8px; background-color: #f6f8fa; height: 140px; font-size: 13px;">
                                <b style="color: #0969da;">[{idx}] {source_file}</b><br>
                                <span style="color: #57606a; font-size: 11px;">SEC Corporate Filing</span><br><br>
                                <span style="color: #0969da; font-weight: 500; font-size: 12px;">🔗 Open Direct Source &rarr;</span>
                            </div>
                        </a>
                        """
            st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Detailed Record & Metadata View")
        for idx, node in enumerate(nodes, start=1):
          source_file = node.metadata.get("file_name", f"Filing_{idx}")
          with st.expander(
              f"📄 Source [{idx}] Details: {source_file}", expanded=(idx == 1)
          ):
            st.markdown(f"**Source File:** `{source_file}`")
            st.markdown("**Retrieved Excerpt & Context:**")
            st.markdown(clean_html(node.text.strip()))
