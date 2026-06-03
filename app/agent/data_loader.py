import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

# Use remote HF embeddings to save memory and storage on Render/Vercel
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    model="BAAI/bge-small-en-v1.5"
)
vector_store = InMemoryVectorStore(embedding=embeddings)

def load_documents():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    cv_path = os.path.join(data_dir, "cv.pdf")
    context_path = os.path.join(data_dir, "context.txt")

    docs = []
    if os.path.exists(cv_path):
        docs.extend(PyPDFLoader(cv_path).load())
    else:
        print(f"Warning: CV not found at {cv_path}")

    if os.path.exists(context_path):
        docs.extend(TextLoader(context_path).load())
    else:
        print(f"Warning: Context file not found at {context_path}")

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # In-memory index
    vector_store.add_documents(splits)
    print(f"Loaded {len(splits)} document chunks into the vector store.")
    return vector_store.as_retriever(search_kwargs={"k": 4})

# Initialize the retriever globally
retriever = load_documents()
