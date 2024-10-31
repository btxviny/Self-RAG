from dotenv import load_dotenv
from langgraph.graph import END, StateGraph



from components.state import GraphState
from components.nodes import (
    route_question_node,
    retrieval_node,
    web_search_node,
    grade_documents_node,
    generation_node,
    grade_answer_node
)
from loguru import logger
load_dotenv()



def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        logger.info("DECISION: Not all documents are relevant to the question, proceed with web search.")
        return "websearch"
    else:
        logger.info("DECISION: All documents are relevant, proceed with answer generation.")
        return "generate"



workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generation_node)
workflow.add_node("websearch", web_search_node)


workflow.set_conditional_entry_point(
    route_question_node,
    {
        "websearch": "websearch",
        "retrieve": "retrieve",
    },
)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_answer_node,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)


app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
