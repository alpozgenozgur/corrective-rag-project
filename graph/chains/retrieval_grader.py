from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()





llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash',
                         temperature = 0)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description='Documents are relevant to the question, yes or no.')

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_prompt= """ You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. """

grade_prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('human', 'Retrieved Document: {document} User question: {question}'),
]

)

retrieval_grader = grade_prompt | structured_llm_grader


