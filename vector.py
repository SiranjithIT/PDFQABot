from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

async def load_pdf(file_path):
  loader = PyPDFLoader(file_path)
  pages = []
  async for page in loader.alazy_load():
      pages.append(page)
  return pages

def text_splitter(pages):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  return text_splitter.split_documents(pages)


def embed_documents(text):
  vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
  )
  vector_store.add_documents(text)
  return vector_store
  
pages = asyncio.run(load_pdf("Digital_News_Report_2023.pdf"))
text = text_splitter(pages)
vector_store = embed_documents(text)
retriever = vector_store.as_retriever()