from typing import Literal
import os

import dotenv
from pydantic import BaseModel, Field

from components.llm import llm
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
dotenv.load_dotenv()


#-------------------------------Generation---------------------------------------------------
rag_prompt = PromptTemplate(
    template="""You are a knowledgeable assistant. 
    Use the information from the following documents to answer the question accurately:
    
    {context}

    Question: {question}
    Answer: """,
    input_variables=["context", "question"]
)

generation_agent = rag_prompt | llm | StrOutputParser()
#--------------------------------Router------------------------------------------------------------
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
system = """
    You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents from a thesis paper on Deep Learning Video Stabilization.
    Use the vectorstore for questions on these topics. For all else, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | llm.with_structured_output(RouteQuery)
#--------------------------------Retrieval Grader--------------------------------------------------
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
system = """
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)
#---------------------------------Answer Grader-----------------------------------------------------
class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
system = """
    You are a grader assessing whether an answer addresses / resolves a question \n 
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
answer_grader: RunnableSequence = answer_prompt | llm.with_structured_output(GradeAnswer)
#---------------------------------Hallucination Grader-----------------------------------------------------
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
system = """
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader: RunnableSequence = hallucination_prompt | llm.with_structured_output(GradeHallucinations)