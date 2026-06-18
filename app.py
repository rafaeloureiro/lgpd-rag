import streamlit as st
import os
from pypdf import PdfReader # <-- Nova biblioteca importada
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ... (Mantenha as configurações iniciais da página e da API Key)

@st.cache_resource
def inicializar_base_conhecimento():
    # Carrega o modelo de embedding local
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Verifica se o arquivo PDF existe
    if not os.path.exists("lgpd.pdf"):
        st.error("Erro: O arquivo 'lgpd.pdf' não foi encontrado na raiz do projeto.")
        st.stop()
        
    # Lendo o arquivo PDF e extraindo o texto
    reader = PdfReader("lgpd.pdf")
    texto_completo = ""
    for page in reader.pages:
        texto_pag = page.extract_text()
        if texto_pag:
            texto_completo += texto_pag + "\n"
    
    # Divide o texto em blocos (chunks) baseados em quebras de linha
    chunks = [c.strip() for c in texto_completo.split("\n") if len(c.strip()) > 30]
    
    # Gerar embeddings dos chunks e criar o índice FAISS
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return model, index, chunks

# ... (O restante do código do chat e chamada da API Groq continua exatamente igual)
