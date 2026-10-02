import os
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sec_edgar_downloader import Downloader

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = None

dl = Downloader("MyFintechOrg", "testuser@myfintech.com", "sec_filings")
dl.get("10-K", "AAPL", limit=1, download_details=True)

documents = []
base_dir = "sec_filings/sec-edgar-filings"

if os.path.exists(base_dir):
  for root, dirs, files in os.walk(base_dir):
    for file in files:
      if file.endswith(".txt") or file.endswith(".html"):
        file_path = os.path.join(root, file)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
          text_content = f.read()

        path_parts = root.split(os.sep)
        sec_url = "https://www.sec.gov/edgar/searchedgar/companysearch"
        if len(path_parts) >= 4:
          ticker = path_parts[2]
          sec_url = f"https://www.sec.gov/edgar/browse/?CIK={ticker}"

        doc = Document(
            text=text_content,
            metadata={
                "file_name": file,
                "source_url": sec_url,
            },
        )
        documents.append(doc)

index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./storage")
