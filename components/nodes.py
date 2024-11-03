from typing import Any, Dict

from components.state import GraphState
from components.vector_store import retriever
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from loguru import logger
from components.agents import (
    answer_grader,
    generation_agent,
    hallucination_grader,
    question_router,
    retrieval_grader
) 

web_search_tool = TavilySearchResults(k=3)

def route_question_node(state: GraphState) -> str:
    """
    Routes a question to the appropriate data source based on the question's content.
    This function logs debug information about the routing process and determines
    whether the question should be routed to a web search or a vector store for retrieval.
    Args:
        state (GraphState): The current state containing the question to be routed.
    Returns:
        str: A string indicating the data source to which the question is routed.
             Possible values are "websearch" for web search and "retrieve" for vector store.
    """

    question = state["question"]
    logger.info("Routing question: {}", question)

    source = question_router.invoke({"question": question})
    if source.datasource == "websearch":
        logger.info("Routing question to web search.")
        return "websearch"
    elif source.datasource == "vectorstore":
        logger.info("Routing question to vector store (RAG).")
        return "retrieve"

def retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves documents based on the provided question in the state.
    Args:
        state (GraphState): The current state of the graph, which includes the question to be processed.
    Returns:
        Dict[str, Any]: A dictionary containing the retrieved documents and the original question.
    """

    question = state["question"]
    logger.info("Retrieving documents...")

    documents = retriever.invoke(question)
    logger.info("Retrieved {} documents.", len(documents))
    return {"documents": documents, "question": question}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    """
    Perform a web search based on the provided question and update the state with the search results.
    Args:
        state (GraphState): The current state containing the question and documents.
    Returns:
        Dict[str, Any]: The updated state with the web search results appended to the documents.
    """

    question = state.get("question")
    documents = state.get("documents")
    logger.info("Performing web search for question: {}", question)

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results_doc = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results_doc)
    else:
        documents = [web_results_doc]
    
    logger.info("Appended web search results. Total documents: {}", len(documents))
    return {"documents": documents, "question": question}

def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant, it sets a flag to trigger a web search.
    Args:
        state (GraphState): The current graph state.
    Returns:
        Dict[str, Any]: The updated state with relevant documents and a web search flag if necessary.
    """

    question = state.get("question")
    documents = state.get("documents")
    logger.info("Grading relevance of documents for question...")

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        if score.binary_score.lower() == "yes":
            logger.info("Document {} relevant to question.",d.metadata)
            filtered_docs.append(d)
        else:
            logger.info("Document {} not relevant to question;",d.metadata)
            web_search = True

    logger.info("Filtered documents count: {}", len(filtered_docs))
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def generation_node(state: GraphState) -> Dict[str, Any]:
    """
    Processes the given state to generate a response based on the provided question and documents.
    Args:
        state (GraphState): The current state with "question" and "documents".
    Returns:
        Dict[str, Any]: A dictionary containing the original "documents", "question", and the generated response.
    """

    question = state.get("question")
    documents = state.get("documents")
    logger.info("Generating response ...")

    generation = generation_agent.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_answer_node(state: GraphState) -> str:
    """
    Grades the generated answer based on its relevance to the provided documents and question.
    Args:
        state (GraphState): A dictionary-like object with "question", "documents", and "generation".
    Returns:
        str: The grade of the generated answer: "useful", "not useful", or "not supported".
    """

    question = state.get("question")
    documents = state.get("documents")
    generation = state.get("generation")
    logger.info("Grading generated answer ...")

    hallucination_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_score.binary_score is not None:
        logger.info("Answer grounded in documents. Checking relevance to question.")
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        
        if answer_score.binary_score is not None:
            logger.info("Generated answer is useful and addresses the question.")
            return "useful"
        else:
            logger.info("Generated answer does not fully address the question.")
            return "not useful"
    else:
        logger.warning("Generated answer is not grounded in documents. Re-evaluation needed.")
        return "not supported"
