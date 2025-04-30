# state_manager.py (Change 1 - Initialize Pinecone VS)

import logging
import streamlit as st

# Adjust the import based on your project structure
from llm.graph_logic import build_graph
from llm.components import get_pinecone_vectorstore

# Use consistent application logger name
logger = logging.getLogger("RAG_Chatbot_App")


def initialize_session_state():
    """Initializes necessary values in Streamlit's session state,
       including the persistent Pinecone vectorstore connection."""

    # Initialize basic states if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.debug("Initialized 'messages' in session state.")
    if "processed_sources" not in st.session_state:
        st.session_state.processed_sources = []
        logger.debug("Initialized 'processed_sources' in session state.")
    if "mode_internal" not in st.session_state:
        st.session_state.mode_internal = "Web Search" # Default mode
        logger.debug("Initialized default chatbot mode (internal): Web Search")

    # --- Initialize Pinecone Vectorstore Connection ---
    if "vectorstore" not in st.session_state:
        logger.info("Attempting to initialize Pinecone vectorstore connection...")
        index_name = st.session_state.get("PINECONE_INDEX_NAME") # Get index name loaded earlier
        if index_name:
            # Call the function to get the vectorstore object
            vs = get_pinecone_vectorstore(index_name)
            if vs:
                st.session_state.vectorstore = vs # Store the connection object
                logger.info("Pinecone vectorstore initialized and stored in session state.")
            else:
                st.session_state.vectorstore = None # Explicitly set to None if connection failed
                logger.error("Failed to initialize Pinecone vectorstore during session state init.")
                # Optionally show a warning in the UI, but app can continue in Web Search mode
                # st.warning("Could not connect to the RAG vector database. RAG mode may not function correctly.")
        else:
            st.session_state.vectorstore = None # Set to None if index name is missing
            logger.error("Cannot initialize Pinecone vectorstore: PINECONE_INDEX_NAME missing from session state.")
            # st.error("Configuration Error: Cannot initialize RAG database - Index Name missing.")
    # ---

    # Initialize Graph (needs to happen after other states are potentially set)
    if "graph" not in st.session_state:
        logger.info("Compiling graph for the first time...")
        try:
            st.session_state.graph = build_graph()
            logger.info("Compiled and saved graph to session state.")
        except Exception as e:
            logger.error(f"Critical error compiling graph: {e}", exc_info=True)
            st.error(f"Cannot initialize application due to graph compilation error: {e}")
            st.stop()

