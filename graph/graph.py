from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from graph.node_constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()

def decide_to_generate(state: GraphState):
    print('---ASSES GRADED DOCUMENTS---')
    if state['web_search']:
        print('WEBSEARCH')
        return WEBSEARCH
    else:
        return GENERATE

def grade_generation_grounded_in_documents(state: GraphState):
    print('CHECK HALLUCINATION')
    question=state['question']
    documents=state['documents']
    generation=state['generation']

    score= hallucination_grader.invoke(
        {'documents': documents, 'generation': generation}
    )
    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState):
    print('route question')
    question=state['question']
    source = question_router.invoke({'question': question})
    if source.datasource == WEBSEARCH:
        print('WEBSEARCH')
        return WEBSEARCH
    elif source.datasource == 'vectorstore':
        return RETRIEVE




workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)
workflow.set_conditional_entry_point(
    route_question,
    {WEBSEARCH:WEBSEARCH,
     RETRIEVE:RETRIEVE},

)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH:WEBSEARCH,
     GENERATE:GENERATE,}
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents,
    {'not supported': GENERATE,
     'useful':END,
     'not useful':WEBSEARCH}
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)




app = workflow.compile()
