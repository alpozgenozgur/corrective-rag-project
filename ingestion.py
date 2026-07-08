import time

from dotenv import load_dotenv
from google.genai.errors import ClientError
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

# Free-tier embedding quota is consumed per text sent in a request, not per
# request, so a single ~100-document batch can exhaust it in one call. Keep
# batches small and pace them so we stay under the per-minute quota.
EMBED_BATCH_SIZE = 20
EMBED_BATCH_DELAY_SECONDS = 20


def ingest():
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs_list)

    store = Chroma(
        collection_name='rag-chroma',
        embedding_function=embeddings,
        persist_directory='./.chroma',
    )

    for i in range(0, len(splits), EMBED_BATCH_SIZE):
        batch = splits[i:i + EMBED_BATCH_SIZE]
        while True:
            try:
                store.add_documents(batch)
                break
            except ClientError as e:
                if e.code != 429:
                    raise
                print(f'Quota hit on batch {i}, waiting {EMBED_BATCH_DELAY_SECONDS}s before retrying...')
                time.sleep(EMBED_BATCH_DELAY_SECONDS)
        if i + EMBED_BATCH_SIZE < len(splits):
            time.sleep(EMBED_BATCH_DELAY_SECONDS)

    return store


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
