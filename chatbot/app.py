import logging
from typing import Any, Dict, Generator, List

import streamlit as st

try:
    from chatbot.config import settings, setup_logging
    from chatbot.processing import process_rag_sources
    from chatbot.state_manager import (
        initialize_session_state,
    )
    from chatbot.ui_chat import display_ai_message_elements, display_chat_history
    from chatbot.ui_sidebar import display_sidebar
    from llm.graph_logic import GraphState
except ImportError as e:
    st.error(f"Failed to import necessary project modules: {e}. Please check paths.")
    st.stop()


logger = setup_logging()


if "PINECONE_INDEX_NAME" not in st.session_state:
    if hasattr(settings, "PINECONE_INDEX_NAME") and settings.PINECONE_INDEX_NAME:
        st.session_state.PINECONE_INDEX_NAME = settings.PINECONE_INDEX_NAME
    else:
        st.error("Critical Error: Pinecone index name not configured in settings.")
        st.stop()

initialize_session_state()


st.set_page_config(page_title="Versatile Chatbot", layout="wide")
st.title("Versatile Chatbot (RAG / Web Search)")

css = r"""
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
"""

st.markdown(css, unsafe_allow_html=True)


uploaded_files, urls, process_rag_button = display_sidebar()

if process_rag_button:
    if uploaded_files is not None or urls:
        process_rag_sources(uploaded_files, urls)
    else:
        logger.warning("Process RAG button clicked, but RAG sources are not available.")


current_mode_internal = st.session_state.get("mode_internal", "Web Search")
display_mode = "Web Search (Tavily)" if current_mode_internal == "Web Search" else "RAG"
st.markdown(f"**Current Mode:** {display_mode}")

display_chat_history()


if prompt := st.chat_input("Enter your question here..."):
    logger.info(
        f"Received question (Mode: {current_mode_internal}, Length: {len(prompt)})."
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("ai", avatar="🤖"):

        response_placeholder, status_list_placeholder = display_ai_message_elements()

        if "graph" not in st.session_state or st.session_state.graph is None:
            st.error("Chat logic graph is not initialized.")
            logger.error("Graph not found in session state.")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Error: Chat logic graph is not initialized.",
                    "steps": ["❌ Initialization Error"],
                }
            )
            st.rerun()
        else:
            compiled_graph = st.session_state.graph
            full_response = ""
            graph_error = None

            final_run_status_history = ["Processing started..."]

            try:
                graph_input: GraphState = {
                    "query": prompt,
                    "chat_history": st.session_state.messages[:-1],
                    "chatbot_mode": current_mode_internal,
                    "use_rag": st.session_state.get("vectorstore") is not None,
                    "context": None,
                    "generation": None,
                    "generation_stream": None,
                    "status_message": "Processing started...",
                    "web_search_sources": None,
                    "rag_chunk_ids": None,
                    "rag_chunk_scores": None,
                    "run_status_history": ["Processing started..."],
                }
                logger.debug(
                    f"Graph input prepared (excluding history): { {k: v for k, v in graph_input.items() if k != 'chat_history'} }"
                )

                events: Generator[Dict[str, Any], None, None] = compiled_graph.stream(
                    graph_input, config={"recursion_limit": 10}
                )

                latest_full_history_from_graph = final_run_status_history[:]

                for event in events:
                    logger.debug(f"Graph event: {event}")
                    event_keys = list(event.keys())
                    if not event_keys or event_keys[0] == "__end__":
                        continue
                    node_name = event_keys[0]
                    node_output = event.get(node_name)

                    current_status_display_list = []
                    if isinstance(node_output, dict):

                        latest_full_history_from_graph = node_output.get(
                            "run_status_history", latest_full_history_from_graph
                        )
                        current_status_display_list = latest_full_history_from_graph

                        if node_name in ["generate", "inform_no_rag_data"]:
                            generation_stream = node_output.get("generation_stream")
                            if generation_stream and hasattr(
                                generation_stream, "__iter__"
                            ):
                                try:
                                    for chunk in generation_stream:

                                        chunk_content = getattr(
                                            chunk,
                                            "content",
                                            chunk if isinstance(chunk, str) else None,
                                        )
                                        if chunk_content:
                                            full_response += chunk_content
                                            response_placeholder.markdown(
                                                full_response + "▌"
                                            )
                                except Exception as stream_ex:
                                    logger.error(
                                        f"Error iterating stream from '{node_name}': {stream_ex}",
                                        exc_info=True,
                                    )
                                    graph_error = stream_ex

                                    current_status_display_list.append(
                                        f"❌ Error reading AI stream: {stream_ex}"
                                    )

                    status_markdown = "\n".join(
                        [f"- {s}" for s in current_status_display_list]
                    )
                    status_list_placeholder.markdown(status_markdown)

                final_run_status_history = latest_full_history_from_graph

                if graph_error:
                    final_status_msg = f"❌ Error: {graph_error}"
                    if not full_response:
                        full_response = (
                            f"Sorry, an error occurred during processing: {graph_error}"
                        )
                else:
                    final_status_msg = "✅ Completed!"
                    if not full_response:
                        full_response = "Sorry, I couldn't generate a response based on the information found."

                if (
                    not final_run_status_history
                    or final_status_msg != final_run_status_history[-1]
                ):
                    final_run_status_history.append(final_status_msg)

                status_markdown = "\n".join(
                    [f"- {s}" for s in final_run_status_history]
                )
                status_list_placeholder.markdown(status_markdown)
                response_placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "steps": final_run_status_history,
                    }
                )

            except Exception as e:
                graph_error = e
                error_message = f"Critical error running chat logic: {graph_error}"
                logger.error(error_message, exc_info=True)

                final_run_status_history.append(f"❌ {error_message}")

                full_response = f"Sorry, a critical error occurred: {graph_error}"
                response_placeholder.markdown(full_response)
                status_markdown = "\n".join(
                    [f"- {s}" for s in final_run_status_history]
                )
                status_list_placeholder.markdown(status_markdown)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "steps": final_run_status_history,
                    }
                )
