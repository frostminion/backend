from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

def load_popu_chain():
    os.environ["OPENAI_API_KEY"] = "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"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vector_index_popu/popu_index", embeddings, allow_dangerous_deserialization=True)

    prompt_template = """Halo! Kamu asisten digital Telkomsel yang siap bantu rekomendasi paket internet.

    Berdasarkan info berikut:

    {context}

    Perlu diingat, tidak semua penduduk dengan penghasilan tinggi atau yang tinggal di wilayah dengan jumlah penduduk 
    besar menginginkan paket internet yang mahal. Seringkali, justru paket internet yang lebih terjangkau lebih sesuai dengan kebutuhan mereka. 
    Sebab, kebutuhan layanan internet setiap kelompok berbeda-beda, tergantung pada kebiasaan dan kecenderungan penggunaan sehari-hari.
    Jadi, sesuaikan rekomendasinya dengan kebiasaan dan kebutuhan pelanggan di daerah itu, ya!

    Pertanyaan: {question}

    Berikan jawaban yang jelas dan gampang dipahami dari data di atas."""

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
