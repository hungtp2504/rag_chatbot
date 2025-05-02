import logging

import streamlit as st

from llm.components import (
    embed_and_upsert,
    load_and_split,
)

logger = logging.getLogger("RAG_Chatbot_App")


def process_rag_sources(uploaded_files, urls):
    """Handles the processing of RAG data sources (files and URLs)."""
    logger.info("Starting RAG source processing workflow...")
    sources_to_process = []
    source_names = []

    if uploaded_files:
        for file in uploaded_files:
            sources_to_process.append(file)
            source_names.append(f"PDF: {file.name}")
        logger.info(f"Received {len(uploaded_files)} PDF files.")

    if urls:
        url_list = [url.strip() for url in urls.split("\n") if url.strip()]
        sources_to_process.extend(url_list)
        source_names.extend([f"URL: {url}" for url in url_list])
        logger.info(f"Received {len(url_list)} URLs.")

    if sources_to_process:
        logger.info(f"Starting processing for {len(source_names)} RAG sources.")

        with st.spinner("🔄 Processing RAG data sources..."):
            try:

                docs = load_and_split(sources_to_process)

                if docs:
                    logger.info("Initializing vectorstore for RAG...")
                    vectorstore = st.session_state.vectorstore

                    if vectorstore:

                        st.session_state.vectorstore = vectorstore
                        logger.info("RAG Vectorstore is ready.")

                        success = embed_and_upsert(docs, st.session_state.vectorstore)
                        if success:

                            st.session_state.processed_sources = source_names
                            logger.info(
                                f"Successfully processed {len(source_names)} RAG sources."
                            )

                            st.success(
                                f"✅ Successfully processed {len(source_names)} RAG sources!"
                            )
                            st.rerun()
                        else:

                            logger.error("Error during RAG data embed/upsert.")
                            st.error("❌ Embed/upsert RAG error.")
                    else:

                        logger.error("Could not connect to Pinecone Index for RAG.")
                        st.error("❌ Pinecone RAG connection error.")
                else:

                    logger.warning(
                        "No content extracted from the provided RAG sources."
                    )
                    st.warning("⚠️ No RAG content extracted.")
            except Exception as e:

                logger.error(f"Error during RAG source processing: {e}", exc_info=True)
                st.error(f"❌ Error processing RAG sources: {e}")
    else:

        logger.warning("No RAG sources provided for processing.")
        st.warning("⚠️ Please provide RAG data sources to process.")
