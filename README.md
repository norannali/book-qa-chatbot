# 🤖 AI-Powered Book QA Chatbot – Telegram Bot (Hands-On ML Edition)

An intelligent Telegram chatbot that answers your questions based on the book:

📘 **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**  
by Aurélien Géron (2nd or 3rd Edition)

This project demonstrates how to use **LLMs + Retrieval-Augmented Generation (RAG)** to make books interactive and queryable — just like chatting with the author!

---

## 🔍 Use Case

Ever wished you could **ask a textbook questions** like:

- "What's the difference between Batch Gradient Descent and Stochastic?"
- "Explain regularization techniques in Chapter 4."
- "What is the role of activation functions in deep learning?"

Now you can — directly on Telegram 📱

---

## 🚀 Features

- 📖 Ask any question based on *Hands-On ML*
- 💬 Get instant replies via a Telegram bot
- 🧠 Context-aware answers powered by local LLM (Ollama)
- 📚 Uses vector search (FAISS) to retrieve relevant book chunks
- 🔐 Fully local – no data sent to external APIs

---

## 📌 Architecture

PDF Book ➜ Text Splitter ➜ Embeddings (Ollama) ➜ FAISS Vector Store

⬇ ⬇


User Question ➜ RAG QA Chain ➜ LLM Answer via Telegram Bot

---

## 🛠️ Installation

### 1. Clone the Repo

```bash
git clone https://github.com/norannali/book-qa-chatbot
cd book-qa-telegram-bot
```
---
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
---
### 3. Run Ollama (LLM backend)
 - Make sure Ollama is installed and that you're running a model like llama3:
```bash
ollama run llama3
```
---
## 🛠 How It Works

1. The PDF is split into chunks and converted to vector embeddings.
2. Chunks are stored in a FAISS vector database.
3. When a user asks a question via Telegram:
   - Top relevant chunks are retrieved.
   - A prompt is built and sent to the LLM.
   - The LLM generates a context-aware answer.
   - The answer is sent back to the user via Telegram.

---

## 🔧 Future Enhancements

- [ ] Add memory and multi-turn conversation  
- [ ] Add support for multiple books  
- [ ] Dockerize the app for easier deployment  
- [ ] Build a Streamlit or Gradio UI (optional)  
- [ ] Add voice-based input/output via Telegram  

---

## 📚 Tech Stack

- **[LangChain](https://www.langchain.com/)** – LLM pipeline and QA chain  
- **[FAISS](https://github.com/facebookresearch/faiss)** – Vector similarity search  
- **[Ollama](https://ollama.com/)** – Local LLMs like LLaMA3 or Mistral  
- **[python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)** – Telegram bot integration  
- **[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)** or **PDFMiner** – PDF parsing  

---

## 🧠 Recommended Requirements

- Python 3.10+  
- Telegram bot token from [BotFather](https://t.me/BotFather)  
- Ollama installed and running locally  
- A downloaded LLM model (e.g., llama3, mistral, etc.)  
- Legal access to the *Hands-On Machine Learning* PDF  

---

## 🏷 License

MIT License.  
**Educational use only.**  
Please respect the copyright of the book.

---
