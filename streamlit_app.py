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
          sec_url = node.metadata.get(
              "source_url", "https://www.sec.gov/edgar/searchedgar/companysearch"
          )

          with col:
            st.markdown(
                f"""
                        <a href="{sec_url}" target="_blank" style="text-decoration: none; color: inherit;">
                            <div style="padding: 14px; border: 1px solid #d0d7de; border-radius: 8px; background-color: #f6f8fa; height: 140px; font-size: 13px; transition: background-color 0.2s;">
                                <b style="color: #0969da;">[{idx}] {source_file}</b><br>
                                <span style="color: #57606a; font-size: 11px;">SEC Corporate Filing</span><br><br>
                                <span style="color: #0969da; font-weight: 500; font-size: 12px;">🔗 Open Direct Source &rarr;</span>
                            </div>
                        </a>
                        """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Detailed Record & Metadata View")
        for idx, node in enumerate(nodes, start=1):
          source_file = node.metadata.get("file_name", f"Filing_{idx}")
          sec_url = node.metadata.get(
              "source_url", "https://www.sec.gov/edgar/searchedgar/companysearch"
          )

          with st.expander(
              f"📄 Source [{idx}] Details: {source_file}", expanded=(idx == 1)
          ):
            col_meta1, col_meta2 = st.columns([1, 2])
            with col_meta1:
              st.markdown("**Document Type:** SEC 10-K / 10-Q")
              st.markdown(f"**Source File:** `{source_file}`")
              st.markdown(
                  f"🔗 **[Open Direct Filing URL]({sec_url})**",
                  unsafe_allow_html=True,
              )
            with col_meta2:
              st.markdown("**Retrieved Excerpt & Context:**")
              st.markdown(clean_html(node.text.strip()))

        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

        st.markdown("### Overview of disclosures")
        for idx, node in enumerate(nodes, start=1):
          st.markdown(f"**[{idx}]** {clean_html(node.text.strip())}")
          st.markdown("")

        col1, col2 = st.columns([4, 1])
        with col2:
          st.markdown(
              """
                    <div style="display: flex; gap: 12px; justify-content: flex-end; align-items: center; font-size: 14px; color: #555;">
                        <span style="cursor: pointer;" title="Thumbs Up">👍</span>
                        <span style="cursor: pointer;" title="Thumbs Down">👎</span>
                        <span style="cursor: pointer;" title="Copy text">📋 Copy</span>
                        <span style="cursor: pointer;" title="Regenerate">🔄 Try again</span>
                    </div>
                    """,
              unsafe_allow_html=True,
          )

        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

        st.markdown("**Related research questions**")
        q_col1, q_col2 = st.columns(2)
        with q_col1:
          if st.button(
              "🔍 How does management evaluate credit and liquidity risk?"
          ):
            st.info("Triggering query...")
        with q_col2:
          if st.button("🔍 What are the primary risk factors disclosed?"):
            st.info("Triggering query...")
  else:
    st.warning("Please enter a question to search.")
