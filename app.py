import streamlit as tf
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configuração da página do Streamlit
st.set_page_config(page_title="RAG - Especialista LGPD", page_icon="⚖️", layout="centered")
st.title("⚖️ Assistente Virtual da LGPD (RAG)")
st.write("Faça perguntas sobre a Lei Geral de Proteção de Dados baseadas diretamente no texto da lei.")

# Chave da API do Groq (Recomendável usar o st.secrets no deploy)
GROQ_API_KEY = st.sidebar.text_input("Insira sua Groq API Key", type="password")

if not GROQ_API_KEY:
    st.info("Por favor, adicione sua Groq API Key na barra lateral para começar.", icon="🔑")
    st.stop()

# Inicializa o cliente Groq
client = Groq(api_key=GROQ_API_KEY)

# 1. Carregar e Processar o Documento (Cache para não reprocessar a cada clique)
@st.cache_resource
def inicializar_base_conhecimento():
    # Carrega o modelo de embedding (roda localmente e é leve)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Verifica se o arquivo da LGPD existe
    if not os.path.exists("lgpd.txt"):
        # Criando um arquivo dummy caso não exista, substitua pelo real!
        with open("lgpd.txt", "w", encoding="utf-8") as f:
            f.write("Art. 1º Esta Lei dispõe sobre o tratamento de dados pessoais, inclusive nos meios digitais...\n"
                    "Art. 2º São fundamentos da disciplina da proteção de dados pessoais: I - o respeito à privacidade...")
            
    with open("lgpd.txt", "r", encoding="utf-8") as f:
        texto_completo = f.read()
    
    # Chunking simples: divide por parágrafos/linhas vazias
    chunks = [c.strip() for c in texto_completo.split("\n") if len(c.strip()) > 20]
    
    # Gerar embeddings dos chunks
    embeddings = model.encode(chunks)
    
    # Criar o índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return model, index, chunks

with st.spinner("Carregando base de conhecimento da LGPD..."):
    embed_model, vector_index, txt_chunks = inicializar_base_conhecimento()

# 2. Interface de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do usuário
if user_query := st.chat_input("Ex: Quais são os direitos do titular dos dados?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Buscando na lei e gerando resposta..."):
            # A. Recuperação (Retrieval)
            query_embedding = embed_model.encode([user_query])
            D, I = vector_index.search(np.array(query_embedding).astype('float32'), k=3) # Recupera os 3 trechos mais relevantes
            
            contexto = "\n".join([txt_chunks[idx] for idx in I[0] if idx < len(txt_chunks)])
            
            # B. Geração (Augmentation & Generation)
            prompt_sistema = (
                "Você é um advogado especialista em LGPD (Lei Geral de Proteção de Dados do Brasil).\n"
                "Responda à pergunta do usuário utilizando APENAS o contexto fornecido abaixo extraído da lei.\n"
                "Se a resposta não puder ser encontrada no contexto, diga educadamente que não encontrou essa informação na base da lei.\n\n"
                f"CONTEXTO DA LEI:\n{contexto}"
            )
            
            # Chamada para a Groq API
            completion = client.chat.completions.create(
                model="llama3-8b-8192", # Modelo rápido e eficiente
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.2, # Baixa temperatura para evitar alucinações
            )
            
            resposta = completion.choices[0].message.content
            st.markdown(resposta)
            
            # Opcional: Mostrar as fontes embaixo da resposta
            with st.expander("🔍 Ver trechos da lei utilizados como base"):
                st.write(contexto)
                
    st.session_state.messages.append({"role": "assistant", "content": resposta})
