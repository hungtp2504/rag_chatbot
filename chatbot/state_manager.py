import logging

import streamlit as st

try:
    from chatbot.config import settings
    from llm.components import get_pinecone_vectorstore
    from llm.graph_logic import build_graph
except ImportError:
    from config import settings

    from llm.components import get_pinecone_vectorstore
    from llm.graph_logic import build_graph


logger = logging.getLogger("RAG_Chatbot_App")


def initialize_session_state():
    """Initializes necessary values in Streamlit's session state,
    including the persistent Pinecone vectorstore connection."""

    if "PINECONE_INDEX_NAME" not in st.session_state:
        if hasattr(settings, "PINECONE_INDEX_NAME") and settings.PINECONE_INDEX_NAME:
            st.session_state.PINECONE_INDEX_NAME = settings.PINECONE_INDEX_NAME
            logger.info(
                f"Initialized PINECONE_INDEX_NAME in session state: {settings.PINECONE_INDEX_NAME}"
            )
        else:
            logger.critical(
                "Pinecone index name is not configured in settings and is missing from session state."
            )
            st.error(
                "Lỗi nghiêm trọng: Tên Pinecone index chưa được cấu hình. Vui lòng kiểm tra file .env và đảm bảo PINECONE_INDEX_NAME đã được đặt."
            )
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.debug("Initialized 'messages' in session state.")
    if "processed_sources" not in st.session_state:
        st.session_state.processed_sources = []
        logger.debug("Initialized 'processed_sources' in session state.")
    if "mode_internal" not in st.session_state:
        st.session_state.mode_internal = "Web Search"
        logger.debug("Initialized default chatbot mode (internal): Web Search")

    if "audio_input_version" not in st.session_state:
        st.session_state.audio_input_version = 0
        logger.debug("Initialized 'audio_input_version' for st.audio_input key.")

    if "session_notes" not in st.session_state:
        st.session_state.session_notes = []
        logger.debug("Initialized 'session_notes' list in session state.")

    if "vectorstore" not in st.session_state:
        logger.info("Attempting to initialize Pinecone vectorstore connection...")
        index_name = st.session_state.get("PINECONE_INDEX_NAME")

        vs = get_pinecone_vectorstore(index_name)
        if vs:
            st.session_state.vectorstore = vs
            logger.info("Pinecone vectorstore initialized and stored in session state.")
        else:
            st.session_state.vectorstore = None
            logger.error(
                "Failed to initialize Pinecone vectorstore during session state init."
            )

    if "graph" not in st.session_state:
        logger.info("Compiling graph for the first time...")
        try:
            st.session_state.graph = build_graph()
            logger.info("Compiled and saved graph to session state.")
        except Exception as e:
            logger.error(f"Critical error compiling graph: {e}", exc_info=True)
            st.error(f"Không thể khởi tạo ứng dụng do lỗi biên dịch đồ thị: {e}")
            st.stop()
