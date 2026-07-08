from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

urls = ['https://lilianweng.github.io/posts/2023-06-23-agent/',
        'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/',
        'https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/'
        ]

embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')


def ingest():
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs_list)

    return Chroma.from_documents(
        documents=splits,
        collection_name='rag-chroma',
        embedding=embeddings,
        persist_directory='./.chroma'
    )


vectorstore = Chroma(
    collection_name='rag-chroma',
    persist_directory='./.chroma',
    embedding_function=embeddings,
)

# Only scrape + embed the source pages once; later runs reuse the persisted collection.
if vectorstore._collection.count() == 0:
    vectorstore = ingest()

retriever = vectorstore.as_retriever()

if __name__ == "__main__":
    vectorstore = ingest()
    retriever = vectorstore.as_retriever()
