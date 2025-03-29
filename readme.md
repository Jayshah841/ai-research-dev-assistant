# LangGraph + DeepSeek-R1 + Function Call + Agentic RAG (Insane Results)

## Overview

This project is an AI-powered assistant designed to retrieve and analyze research and development data. It uses the **LangChain** framework for text retrieval and embeddings, **ChromaDB** for vector storage, and **Ollama** for AI-based text generation and embeddings.

## Features

- Retrieves information from research and development databases using **vector embeddings**.
- Uses **Ollama** for AI-powered responses.
- **DeepSeek-R1** model for answering questions.
- **Nomic-Embed-Text** for generating document embeddings.
- Provides answers based on retrieved documents or rewrites queries for better results.
- Streamlit-based user interface for easy interaction.

## Installation

### Prerequisites

- Python 3.13.0 or higher
- [Ollama](https://ollama.com/) installed on your system
- Required Python libraries

### Steps

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set up a virtual environment** (optional but recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download and set up Ollama models**

   ```bash
   ollama pull deepseek-r1
   ollama pull nomic-embed-text
   ```

## Usage

### Running the application

1. **Start Ollama** in the background (if not already running):

   ```bash
   ollama run deepseek-r1
   ```

2. **Run the application**

   ```bash
   streamlit run main.py
   ```

## Project Structure

```
📁 Project Root
│── app.py             # Core AI processing and retrieval logic
│── main.py            # Streamlit UI for user interaction
│── requirements.txt   # Required Python dependencies
│── README.md          # Documentation
```

## How It Works

1. **Text Embeddings**

   - The project uses `nomic-embed-text` for embedding research and development texts.
   - These embeddings are stored in **ChromaDB** for fast retrieval.

2. **Retrieval Augmented Generation (RAG) Approach**

   - When a user asks a question, the system determines whether it's research-related or development-related.
   - Based on this classification, it queries the corresponding vector database and retrieves relevant documents.

3. **AI Model for Answers**
   - Uses `deepseek-r1` for answering user queries.
   - If relevant documents are found, it summarizes and provides insights.
   - If no documents match, it rewrites the query for better retrieval.

## Dependencies

- `langchain`
- `streamlit`
- `requests`
- `ollama`
- `chromadb`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Future Enhancements

- Improve query classification using additional NLP techniques.
- Expand the vector database with more research papers.
- Optimize embeddings for better accuracy.

## License

This project is licensed under [MIT License](LICENSE).
