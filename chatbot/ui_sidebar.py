import logging

import streamlit as st

logger = logging.getLogger("RAG_Chatbot_App")


def update_chatbot_mode():
    """Callback function called when the user changes the chatbot mode."""

    selected_display_mode = st.session_state.chatbot_mode_selector
    logger.debug(
        f"Callback update_chatbot_mode - Radio value selected: {selected_display_mode}"
    )

    if selected_display_mode == "Web Search (Tavily)":
        st.session_state.mode_internal = "Web Search"
    else:
        st.session_state.mode_internal = "RAG"
    logger.info(
        f"Callback: Chatbot mode (internal) updated to: {st.session_state.mode_internal}"
    )


def display_sidebar():
    """Displays all sidebar content and returns necessary widget values."""
    with st.sidebar:
        st.header("Chatbot Configuration")

        mode_options = ["Web Search (Tavily)", "RAG (Uploaded Documents)"]

        current_mode_index = (
            0
            if st.session_state.get("mode_internal", "Web Search") == "Web Search"
            else 1
        )
        st.radio(
            "Select operating mode:",
            options=mode_options,
            key="chatbot_mode_selector",
            index=current_mode_index,
            on_change=update_chatbot_mode,
        )

        st.markdown("---")
        st.header("RAG Data Sources")
        st.caption("Process sources if you want to use RAG mode.")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )
        urls = st.text_area("Or enter URLs (one per line)", key="url_input")
        process_button = st.button("Process RAG Sources", key="process_sources_btn")

        st.subheader("Processed Sources (for RAG):")
        if st.session_state.get("processed_sources"):
            for src in st.session_state.processed_sources:
                st.write(f"- {src}")
        else:
            st.write("No sources processed yet.")

    return uploaded_files, urls, process_button
