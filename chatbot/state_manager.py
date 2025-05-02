import logging

import streamlit as st

from llm.components import get_pinecone_vectorstore
from llm.graph_logic import build_graph

logger = logging.getLogger("RAG_Chatbot_App")


def initialize_session_state():
    """Initializes necessary values in Streamlit's session state,
    including the persistent Pinecone vectorstore connection."""

    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.debug("Initialized 'messages' in session state.")
    if "processed_sources" not in st.session_state:
        st.session_state.processed_sources = []
        logger.debug("Initialized 'processed_sources' in session state.")
    if "mode_internal" not in st.session_state:
        st.session_state.mode_internal = "Web Search"
        logger.debug("Initialized default chatbot mode (internal): Web Search")

    if "session_notes" not in st.session_state:
        st.session_state.session_notes = []
        logger.debug("Initialized 'session_notes' list in session state.")

    if "vectorstore" not in st.session_state:
        logger.info("Attempting to initialize Pinecone vectorstore connection...")
        index_name = st.session_state.get("PINECONE_INDEX_NAME")
        if index_name:

            vs = get_pinecone_vectorstore(index_name)
            if vs:
                st.session_state.vectorstore = vs
                logger.info(
                    "Pinecone vectorstore initialized and stored in session state."
                )
            else:
                st.session_state.vectorstore = None
                logger.error(
                    "Failed to initialize Pinecone vectorstore during session state init."
                )

        else:
            st.session_state.vectorstore = None
            logger.error(
                "Cannot initialize Pinecone vectorstore: PINECONE_INDEX_NAME missing from session state."
            )

    if "graph" not in st.session_state:
        logger.info("Compiling graph for the first time...")
        try:
            st.session_state.graph = build_graph()
            logger.info("Compiled and saved graph to session state.")
        except Exception as e:
            logger.error(f"Critical error compiling graph: {e}", exc_info=True)
            st.error(
                f"Cannot initialize application due to graph compilation error: {e}"
            )
            st.stop()
