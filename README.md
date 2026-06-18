# ⚖️ LexGroq-LGPD: Assistente Virtual Inteligente para a LGPD

O **LexGroq-LGPD** é um sistema de **RAG (Retrieval-Augmented Generation)** simplificado, projetado para responder a dúvidas complexas sobre a Lei Geral de Proteção de Dados (Lei nº 13.709/2018) utilizando dados extraídos diretamente do texto oficial da lei. 

A aplicação utiliza o ecossistema do **Streamlit** para a interface, **FAISS** para busca vetorial local e a altíssima velocidade da **Groq API** para geração de respostas sem alucinações.

---

## 🛠️ Tecnologias Utilizadas

*   **Python 3.10+**
*   **Streamlit:** Interface web rápida e interativa.
*   **Groq API (LLama 3):** Processamento de linguagem natural ultra-rápido.
*   **Sentence-Transformers (`all-MiniLM-L6-v2`):** Geração de embeddings locais e gratuitos.
*   **FAISS (Facebook AI Similarity Search):** Banco de dados vetorial em memória para busca rápida de contexto.

---

## 🏗️ Como o Sistema Funciona?

1.  **Leitura & Divisão:** O arquivo `lgpd.txt` (texto oficial da lei) é lido e dividido em pequenos blocos de texto (chunks).
2.  **Vetorização (Embeddings):** Cada bloco é transformado em um vetor numérico pelo modelo da Hugging Face e armazenado no FAISS.
3.  **Recuperação (Retrieval):** Quando você faz uma pergunta, o sistema busca os 3 blocos mais relevantes da lei.
4.  **Geração:** O contexto recuperado e a sua pergunta são enviados para o modelo `llama3-8b` na Groq, que formula uma resposta baseada estritamente na lei.

---

## 🔧 Como Executar Localmente

### 1. Clonar o Repositório
```bash
git clone [https://github.com/seu-usuario/lexgroq-lgpd.git](https://github.com/seu-usuario/lexgroq-lgpd.git)
cd lexgroq-lgpd
