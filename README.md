Uzbek AI Document Assistant (Capstone Project)
A complete production-quality production RAG (Retrieval-Augmented Generation) system tailored for the Uzbek language. This assistant can process PDF, DOCX, and TXT documents, answer questions using a LangChain-based agent, and provides a sentiment analysis API.

🚀 Features
Multimodal Document Support: Load and index PDF, DOCX, and TXT files.
Autonomous Agent: A LangChain-powered agent that routes queries between a Document Tool (RAG), Search Tool (Web), and Sentiment Tool.
Local AI Models: Uses Hugging Face transformers and sentence-transformers. No paid APIs required.
Production API: FastAPI-based endpoint for sentiment analysis.
Tracing: Built-in execution tracing to see the logic behind agent decisions.
Uzbek Focused: Prompts and processing optimized for the Uzbek language.
📁 Project Structure
uzbek-document-assistant/
├── capstone/
│   ├── modules/          # Core logic modules (m01 - m15)
│   └── data/             # Sample datasets and knowledge base
├── notebook/             # Jupyter Notebook for demonstration
├── api.py                # FastAPI entry point
├── app.py                # Main application logic
├── requirements.txt      # Project dependencies
└── README.md             # This file
🛠️ Installation
Clone the repository:
git clone <repository-url>
cd uzbek-document-assistant
Set up a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
pip install -r requirements.txt
⚙️ How to Use
1. Running the Sentiment API
Start the FastAPI server to use the /predict endpoint:

python api.py
The API will be available at http://127.0.0.1:8000. You can test the sentiment analysis:

curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text": "Bugun juda yaxshi kun"}'
2. Indexing Documents
Place your .pdf, .docx, or .txt files in a source directory. The DocumentAssistantAgent uses m04_rag.py and m05_vectordb.py to index these into a local FAISS vector store.

3. Running the Jupyter Notebook
Open the notebook to see the agent in action:

jupyter notebook notebook/d16_p15_langchain_agent.ipynb
The notebook demonstrates:

Knowledge base indexing.
Agent creation.
Querying the agent with three different intents (Document, Search, Sentiment).
Reviewing the execution trace.
🤖 Models Used
LLM: google/gemma-1.1-2b-it (or equivalent lightweight multilingual model) via HuggingFacePipeline.
Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
Sentiment: oliverguhr/german-sentiment-bert (Note: In production, use a dedicated Uzbek model or a strong multilingual one like cardiffnlp/twitter-xlm-roberta-base-sentiment).
📜 License
MIT License