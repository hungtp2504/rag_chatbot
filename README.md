# RAG Chatbot (LangGraph - Pinecone - Tavily - Streamlit)

![til](rag_demo.gif)


This project implements a versatile chatbot using **LangChain (specifically LangGraph)**, **Pinecone**, **Tavily**, **OpenAI**, and **Streamlit**. It is designed to operate in two primary modes:

1.  **Retrieval-Augmented Generation (RAG)**: Answers questions based on information retrieved from a knowledge base built from user-provided documents (PDFs) or web URLs (including arXiv papers, direct PDF links, and general web pages). This knowledge base is stored and queried using Pinecone.
2.  **Web Search**: Answers questions based on real-time information retrieved from the web using the Tavily search API.

The application uses LangGraph to manage the conversation flow, routing user queries to the appropriate retrieval mechanism (Pinecone or Tavily) before generating a final answer using an OpenAI language model.



## Features

-   **Dual Modes**: Seamlessly switch between RAG and Web Search modes.
-   **Multiple RAG Sources**: Ingest data from uploaded PDFs, web URLs (HTML, PDF links), and arXiv paper IDs.
-   **LangGraph Orchestration**: Utilizes LangGraph to define and execute the stateful, multi-step reasoning process (retrieve -> generate).
-   **Vector Store Integration**: Uses Pinecone for efficient storage and similarity search of document embeddings in RAG mode.
-   **Web Search Capability**: Leverages Tavily for fetching current information from the internet.
-   **Streaming Responses**: Displays AI-generated answers and processing status updates in real-time.
-   **User-Friendly Interface**: Built with Streamlit for easy interaction, data uploading, and mode selection.

## Technologies Used

-   **LangChain / LangGraph**: For core application logic, component integration, and graph-based state management.
-   **Pinecone**: Vector database for storing and querying document embeddings in RAG mode.
-   **Tavily**: Search API for real-time web information retrieval.
-   **OpenAI API**: Language model (`gpt-3.5-turbo`) for generating responses based on retrieved context.
-   **Streamlit**: Framework for building the interactive web user interface.
-   **Python**: The primary programming language.
-   **Dotenv**: For managing environment variables (API keys, configuration).

## Installation

### Prerequisites

1.  **Python** 3.8+ (based on LangChain/LangGraph compatibility)
2.  **Git** (for cloning the repository)
3.  **API Keys**:
    * OpenAI API Key
    * Pinecone API Key
    * Pinecone Index Name (you need to create an index in Pinecone beforehand)
    * Tavily API Key

### Steps to Install

1.  Clone the repository (replace with your actual repo URL if applicable):

    ```bash
    git clone https://github.com/hungtp2504/rag_chatbot.git
    cd rag_chatbot
    ```

2.  Set up a Python virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Set up your API keys and configuration (`chatbot/config.py` & `.env`):

    Create a file named `.env` in the root directory (`rag_chatbot_project/`) and add your credentials and settings:
    
    The application's behaviour is configured through environment variables loaded via the `.env` file and accessed through the `Settings` class in `chatbot/config.py`.

    -   **API Keys & Index (Required)**:
        * `OPENAI_API_KEY`: Your OpenAI API key.
        * `PINECONE_API_KEY`: Your Pinecone API key.
        * `PINECONE_INDEX_NAME`: The name of your pre-existing Pinecone index.
        * `TAVILY_API_KEY`: Your Tavily Search API key.
    -   **RAG Settings**:
        * `CHUNK_SIZE`: Max characters per document chunk for Pinecone (Default: 500).
        * `CHUNK_OVERLAP`: Character overlap between chunks (Default: 50).
        * `TOP_K`: Number of relevant chunks to retrieve from Pinecone (Default: 3).
        * `PINECONE_BATCH_SIZE`: Number of vectors to upsert to Pinecone in one batch (Default: 200).
    -   **Web Search Settings**:
        * `TAVILY_MAX_RESULTS`: Number of search results to fetch from Tavily (Default: 3).
    -   **LLM Settings**:
        * The OpenAI model is currently hardcoded in `llm/components.py` (`get_llm` function) as `"gpt-3.5-turbo"` with `temperature=0.7`. To change this, modify the `get_llm` function directly. (Consider making these configurable via `.env` in future improvements).
    -   **Logging**:
        * `LOG_LEVEL`: Sets the logging level (e.g., `DEBUG`, `INFO`, `WARNING`). Default: `INFO`.
        * `APP_VERBOSE`: Set to `"true"` to enable detailed logging output to `app.log` in the root directory. Default: `"false"`.
       

## Usage

### Running the App

To start the Streamlit application, navigate to the project's root directory in your terminal (where `main.py` is located) and run:

```bash
python main.py
```

This command executes the `main.py` script, which in turn launches the Streamlit application defined in `chatbot/app.py`. Access the application via the local URL provided in your terminal (usually `http://localhost:8501`).

### Interacting with the Chatbot

1.  **Select Mode**: Use the radio button in the sidebar to choose between "RAG (Uploaded Documents)" and "Web Search (Tavily)".
2.  **Add RAG Sources (if using RAG mode)**:
    * Upload PDF files using the file uploader.
    * Enter web URLs (one per line) in the text area. Supported URLs include direct PDF links, arXiv abstract/PDF pages, and general web pages.
    * Click the "Process RAG Sources" button. Wait for the processing to complete (a success message will appear). The processed sources will be listed in the sidebar.
3.  **Ask Questions**: Type your question into the chat input box at the bottom and press Enter.
4.  **View Response**: The chatbot will retrieve information based on the selected mode and generate an answer, streamed to the chat interface.
5.  **Check Processing Steps**: Expand the "Detailed Processing Steps" section below the AI's response to see status messages from the LangGraph execution.


## Project Structure

-   `rag_chatbot_project/`
    -   `main.py`: Main entry point script that launches the Streamlit app.
    -   `get_project.py`: Utility script to generate the project structure snapshot.
    -   `pyproject.toml`: Build and formatting configurations.
    -   `.env`: (User-created) Stores API keys and configuration variables.
    -   `app.log`: (Generated if `APP_VERBOSE=true`) Detailed log file.
    -   `requirements.txt`: (User-created) Lists Python dependencies.
    -   `llm/`: Contains core language model, RAG, and LangGraph logic.
        -   `components.py`: Functions for data loading, splitting, embedding, vector store interaction, web search, LLM setup, prompts.
        -   `graph_logic.py`: Defines the LangGraph state, nodes, edges, and builds the compiled graph.
    -   `chatbot/`: Contains Streamlit UI components and application flow logic.
        -   `app.py`: The main Streamlit application file.
        -   `config.py`: Loads settings from `.env` and configures logging.
        -   `processing.py`: Handles the RAG source processing workflow triggered by the UI.
        -   `state_manager.py`: Initializes and manages Streamlit session state (including graph and vectorstore).
        -   `ui_chat.py`: UI functions for displaying the chat interface.
        -   `ui_sidebar.py`: UI functions for displaying the sidebar controls.

## License

This project is licensed under the MIT License.

## Acknowledgments

-   **LangChain/LangGraph**: For the powerful framework enabling the application logic.
-   **Pinecone**: For the efficient vector database service.
-   **Tavily**: For the real-time web search API.
-   **OpenAI**: For the underlying language model capabilities.
-   **Streamlit**: For the easy-to-use web application framework.
