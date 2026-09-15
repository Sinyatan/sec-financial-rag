import os
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# Extremely lightweight embedding (runs well under 150MB RAM)
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

PERSIST_DIR = "./storage"

print("Loading SEC filing documents...")
reader = SimpleDirectoryReader(
    input_dir="sec_filings",
    recursive=True,
    required_exts=[".txt", ".htm", ".html"],
)
documents = reader.load_data()

print("Chunking documents...")
parser = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
nodes = parser.get_nodes_from_documents(documents)

print("Building index with FastEmbed...")
index = VectorStoreIndex(nodes)
index.storage_context.persist(persist_dir=PERSIST_DIR)
print("Index created and saved to storage successfully.")