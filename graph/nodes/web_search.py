
from typing import Any, Dict

from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState

web_search_tool = TavilySearchResults(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print('web_search')

    question = state['question']
    documents = state.get('documents')

    docs = web_search_tool.invoke({'query':question})
    web_result = '\n'.join([d['content'] for d in docs])
    web_result = Document(page_content=web_result)

    if documents is not None:
        documents.append(web_result)

    else:
        documents = [web_result]

    return {'documents': documents, 'question': question, 'web_search': False}

