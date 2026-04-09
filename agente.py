from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Cargar PDF
loader = PyPDFLoader("HOMBRE.pdf")
documents = loader.load()

# 2. Dividir texto
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 3. Embeddings y vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding_model)

# 4. LLM con Ollama
llm = OllamaLLM(model="kiwi_kiwi/gemma-4-uncensores:e4b")

# 5. Cadena RAG
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# 6. Pregunta
query = "¿Cules son los personajes de la obra? Respuesta en español"
response = qa_chain.invoke({"query": query})

print("Respuesta:", response)

