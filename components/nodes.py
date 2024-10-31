from typing import Any, Dict

from components.state import GraphState
from components.vector_store import retriever
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

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
    This function prints debug information about the routing process and determines
    whether the question should be routed to a web search or a vector store for retrieval.
    Args:
        state (GraphState): The current state containing the question to be routed.
    Returns:
        str: A string indicating the data source to which the question is routed.
             Possible values are "websearch" for web search and "retrieve" for vector store.
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "retrieve"

def retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves documents based on the provided question in the state.
    Args:
        state (GraphState): The current state of the graph, which includes the question to be processed.
    Returns:
        Dict[str, Any]: A dictionary containing the retrieved documents and the original question.
    """

    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
    
def web_search_node(state: GraphState) -> Dict[str, Any]:
    """
    Perform a web search based on the provided question and update the state with the search results.
    Args:
        state (GraphState): The current state containing the question and documents.
    Returns:
        Dict[str, Any]: The updated state with the web search results appended to the documents.
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def generation_node(state: GraphState) -> Dict[str, Any]:
    """
    Processes the given state to generate a response based on the provided question and documents.
    Args:
        state (GraphState): A dictionary containing the current state with keys "question" and "documents".
    Returns:
        Dict[str, Any]: A dictionary containing the original "documents", "question", and the generated response.
    """

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_agent.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_answer_node(state: GraphState) -> str:
    """
    Grades the generated answer based on its relevance to the provided documents and question.
    This function performs the following steps:
    1. Checks if the generated answer is grounded in the provided documents.
    2. If grounded, it further checks if the generated answer addresses the provided question.
    3. Returns a grade based on the results of these checks.
    Args:
        state (GraphState): A dictionary-like object containing the following keys:
            - "question": The question to be addressed by the generated answer.
            - "documents": The documents that the generated answer should be grounded in.
            - "generation": The generated answer to be graded.
    Returns:
        str: The grade of the generated answer, which can be one of the following:
            - "useful": If the generated answer is grounded in the documents and addresses the question.
            - "not useful": If the generated answer is grounded in the documents but does not address the question.
            - "not supported": If the generated answer is not grounded in the documents.
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if score.binary_score is not None:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score is not None:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
