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
                # print(docs)
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





# # processing.py (Background Processing)

# import logging
# import threading 
# from typing import List, Union

# import streamlit as st
# from streamlit.runtime.uploaded_file_manager import UploadedFile

# from llm.components import (
#     embed_and_upsert,
#     get_pinecone_vectorstore,
#     load_and_split,
# )

# # Use consistent application logger name
# logger = logging.getLogger("RAG_Chatbot_App")

# # --- Background Task Function ---
# def _background_process_sources(sources_to_process: List[Union[str, UploadedFile]], source_names: List[str]):
#     """
#     This function runs in a separate thread to process RAG sources
#     without blocking the main Streamlit app thread.
#     It performs load, split, embed, and upsert to Pinecone.
#     Updates session state upon completion or error.
#     """
#     thread_id = threading.get_ident()
#     logger.info(f"[Thread-{thread_id}] Background RAG processing started for {len(sources_to_process)} sources.")

#     try:
#         # 1. Load and Split
#         logger.info(f"[Thread-{thread_id}] Loading and splitting documents...")
#         docs = load_and_split(sources_to_process)
#         if not docs:
#             logger.warning(f"[Thread-{thread_id}] No documents extracted after loading/splitting.")
#             st.session_state.rag_processing_status = "error"
#             st.session_state.rag_error_message = "No content extracted from sources."
#             # Trigger rerun to update UI (safer than direct UI manipulation)
#             # st.rerun() # Consider if rerun is needed here or rely on user interaction
#             return # Exit thread

#         logger.info(f"[Thread-{thread_id}] Successfully loaded and split into {len(docs)} chunks.")

#         # 2. Initialize Vectorstore Connection (needed for upsert)
#         index_name = st.session_state.get("PINECONE_INDEX_NAME")
#         if not index_name:
#             logger.error(f"[Thread-{thread_id}] PINECONE_INDEX_NAME missing. Cannot proceed with upsert.")
#             st.session_state.rag_processing_status = "error"
#             st.session_state.rag_error_message = "Configuration Error: Pinecone Index Name missing."
#             # st.rerun()
#             return

#         logger.info(f"[Thread-{thread_id}] Connecting to Pinecone index '{index_name}' for upsert...")
#         vectorstore = get_pinecone_vectorstore(index_name)
#         if not vectorstore:
#             logger.error(f"[Thread-{thread_id}] Failed to connect to Pinecone index '{index_name}'.")
#             st.session_state.rag_processing_status = "error"
#             st.session_state.rag_error_message = f"Failed to connect to Pinecone index '{index_name}'."
#             # st.rerun()
#             return

#         # 3. Embed and Upsert
#         logger.info(f"[Thread-{thread_id}] Starting embed and upsert to Pinecone...")
#         success = embed_and_upsert(docs, vectorstore) # This function now contains robust cleaning

#         if success:
#             logger.info(f"[Thread-{thread_id}] Successfully processed and upserted {len(source_names)} sources to Pinecone.")
#             # Update state to indicate completion and store the ready vectorstore
#             st.session_state.rag_processing_status = "completed"
#             st.session_state.processed_sources = source_names
#             st.session_state.vectorstore = vectorstore # Store the ready vectorstore connection
#             st.session_state.rag_error_message = None # Clear any previous error
#         else:
#             logger.error(f"[Thread-{thread_id}] Embed/upsert process failed.")
#             st.session_state.rag_processing_status = "error"
#             st.session_state.rag_error_message = "Failed to embed or save data to the vector database."
#             st.session_state.vectorstore = None # Ensure vectorstore is None on error

#     except Exception as e:
#         logger.error(f"[Thread-{thread_id}] Unhandled error during background processing: {e}", exc_info=True)
#         st.session_state.rag_processing_status = "error"
#         st.session_state.rag_error_message = f"An unexpected error occurred: {e}"
#         st.session_state.vectorstore = None # Ensure vectorstore is None on error
#     finally:
#         logger.info(f"[Thread-{thread_id}] Background RAG processing thread finished.")
#         # Trigger a rerun for the main thread to pick up the state changes
#         # This is generally safe as it just schedules a rerun, not direct UI manipulation
#         try:
#              st.rerun()
#         except Exception as rerun_err:
#              # This might happen if called at an awkward time during Streamlit shutdown
#              logger.warning(f"[Thread-{thread_id}] Could not trigger rerun after background task: {rerun_err}")


# # --- Main Function Called by UI ---
# def process_rag_sources(uploaded_files: List[UploadedFile], urls: str):
#     """
#     Handles the request to process RAG sources.
#     Starts the actual processing in a background thread.
#     Updates the UI immediately to indicate processing has started.
#     """
#     logger.info("Processing RAG sources request received.")
#     sources_to_process = []
#     source_names = []

#     # Collect sources (same logic as before)
#     if uploaded_files:
#         for file in uploaded_files: sources_to_process.append(file); source_names.append(f"PDF: {file.name}")
#         logger.info(f"Received {len(uploaded_files)} PDF files.")
#     if urls:
#         url_list = [url.strip() for url in urls.split("\n") if url.strip()]; sources_to_process.extend(url_list); source_names.extend([f"URL: {url}" for url in url_list])
#         logger.info(f"Received {len(url_list)} URLs.")

#     if sources_to_process:
#         # --- Start Background Thread ---
#         logger.info(f"Starting background thread to process {len(source_names)} sources.")
#         # Set status to running *before* starting thread
#         st.session_state.rag_processing_status = "running"
#         st.session_state.rag_error_message = None # Clear previous errors
#         st.session_state.processed_sources = [] # Clear previous sources list while processing

#         # Create and start the thread
#         thread = threading.Thread(
#             target=_background_process_sources,
#             args=(sources_to_process, source_names),
#             daemon=True # Allows main program to exit even if thread is running (optional)
#         )
#         thread.start()

#         # --- Update UI Immediately ---
#         # Don't wait for the thread here.
#         # Show a message indicating background processing.
#         # A rerun is usually triggered automatically by the button click,
#         # or we can force one if needed to ensure the spinner/message shows.
#         logger.info("Background thread started. Main thread continues.")
#         # st.rerun() # Force rerun to update UI immediately if button click doesn't suffice

#     else:
#         logger.warning("No RAG sources provided for processing.")
#         st.warning("⚠️ Please provide RAG data sources to process.") # Show warning in UI

