import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

with open("data/faqs.txt", "r") as f:
    content = f.read().split("\n\n")

docs = [Document(page_content=qa) for qa in content]
splitter = RecursiveCharacterTextSplitter()
split_docs = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
db = Chroma.from_documents(split_docs, embedding, persist_directory="embedding/chroma_db")
db.persist()
