from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_chroma import Chroma


client = chromadb.HttpClient(host='localhost', port=8000)
embeddings= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retriever = Chroma(
    client=client,
    collection_name="thesis",
    embedding_function=embeddings,
).as_retriever()
