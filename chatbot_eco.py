from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

def load_eco_chain():
    os.environ["OPENAI_API_KEY"] = "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"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vector_index_eco/eco_index", embeddings, allow_dangerous_deserialization=True)

    prompt_template = """Anda adalah asisten digital Telkomsel...
    Jawablah pertanyaan pengguna *hanya* berdasarkan informasi berikut:
    
    {context}   
    
    Pertanyaan: {question}
    Jawaban akurat dan lengkap berdasarkan data di atas:"""

    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    llm = ChatOpenAI(
        model_name="meta-llama/llama-4-scout:free",
        openai_api_base="https://openrouter.ai/api/v1"
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 50}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
