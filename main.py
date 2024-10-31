import os
import time
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from loguru import logger

# Load environment variables
load_dotenv()

# Setup loguru logging
logger.add("retrieval_qa.log", format="{time} {level} {message}", level="INFO")

def setup_llm():
    """Initializes the LLM model."""
    logger.info("Setting up Azure LLM model...")
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ['API_BASE'],
        openai_api_version=os.environ['API_VERSION'],
        openai_api_key=os.environ['API_KEY'],
        deployment_name=os.environ['GPT_DEPLOYMENT_NAME'],
        openai_api_type="azure"
    )
    return llm

def setup_vectorstore():
    """Initializes the vectorstore with Chroma and HuggingFace embeddings."""
    logger.info("Setting up vector store...")
    client = chromadb.HttpClient(host='localhost', port=8000)
    embeddings= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        client=client,
        collection_name="thesis",
        embedding_function=embeddings,
    )
    return vectorstore

def setup_retrieval_chain(llm, vectorstore):
    """Sets up the retrieval chain."""
    logger.info("Setting up retrieval chain...")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    return retrieval_chain

def run_retrieval_chain(retrieval_chain, query):
    """Runs the retrieval chain for the given query and logs the time taken."""
    logger.info(f"Processing query: {query}")
    start_time = time.time()
    
    try:
        result = retrieval_chain.invoke(input={"input": query})
        elapsed_time = time.time() - start_time
        logger.info(f"Query processed in {elapsed_time:.2f} seconds")
        logger.info(f"Answer: {result['answer']}")
        return result['answer']
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return "An error occurred while processing your request."

def main():
    """Main function to run continuous input for video stabilization retrieval."""
    llm = setup_llm()
    vectorstore = setup_vectorstore()
    retrieval_chain = setup_retrieval_chain(llm, vectorstore)

    print("You can ask your questions. Type 'exit' to stop.")
    logger.info("System is ready for user queries.")
    
    while True:
        # Accept query input from the user
        query = input("Your question: ")

        # Check for exit condition
        if query.lower() == 'exit':
            logger.info("User terminated the session.")
            break
        
        # Process the query
        run_retrieval_chain(retrieval_chain, query)

if __name__ == "__main__":
    print(f'api key: {os.environ["OPENAI_API_KEY"]}')
    main()
