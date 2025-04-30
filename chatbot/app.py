import logging
from typing import Any, Dict, Generator
import streamlit as st

from chatbot.config import setup_logging, settings
from chatbot.processing import process_rag_sources
from chatbot.state_manager import initialize_session_state
from chatbot.ui_chat import display_chat_history, display_ai_message_elements
from chatbot.ui_sidebar import display_sidebar

from llm.graph_logic import GraphState

logger = setup_logging()


if "PINECONE_INDEX_NAME" not in st.session_state:
    st.session_state.PINECONE_INDEX_NAME = settings.PINECONE_INDEX_NAME

initialize_session_state()

st.set_page_config(page_title="Versatile Chatbot", layout="wide")
st.title("Versatile Chatbot (RAG / Web Search)")


uploaded_files, urls, process_button = display_sidebar()


if process_button:
    process_rag_sources(uploaded_files, urls)

display_mode = (
    "Web Search (Tavily)"
    if st.session_state.mode_internal == "Web Search"
    else "RAG (Uploaded Documents)"
)
st.markdown(f"**Current Mode:** {display_mode}")

display_chat_history()

if prompt := st.chat_input("Enter your question here..."):
    logger.info(
        f"Received question (Mode: {st.session_state.mode_internal}, Length: {len(prompt)})."
    )

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("ai", avatar="🤖"):

        response_placeholder, _, status_list_placeholder = display_ai_message_elements()
        current_run_statuses = []

        vectorstore_is_ready = st.session_state.vectorstore is not None
        chat_history_for_graph = st.session_state.messages[:-1]
        graph_input: GraphState = {
            "query": prompt,
            "chat_history": chat_history_for_graph,
            "chatbot_mode": st.session_state.mode_internal,
            "use_rag": vectorstore_is_ready,
            "context": None,
            "generation": None,
            "generation_stream": None,
            "status_message": "Processing started...",
        }
        logger.debug(
            f"Graph input: mode='{graph_input['chatbot_mode']}', query='{graph_input['query'][:50]}...', vs_ready={vectorstore_is_ready}"
        )

        compiled_graph = st.session_state.graph
        full_response = ""
        graph_error = None

        try:
            initial_status = "Processing started..."
            current_run_statuses.append(initial_status)
            status_list_placeholder.markdown(f"- {initial_status}")

            logger.info(
                f"Starting graph stream execution (Mode: {st.session_state.mode_internal})..."
            )

            events: Generator[Dict[str, Any], None, None] = compiled_graph.stream(
                graph_input, config={"recursion_limit": 10}
            )

            for event in events:
                logger.debug(f"Graph event: {event}")
                event_keys = list(event.keys())

                if not event_keys or event_keys[0] == "__end__":
                    continue

                node_name = event_keys[0]
                logger.debug(f"Processing node: '{node_name}'")
                node_output = event.get(node_name)

                new_status_message = None
                if isinstance(node_output, dict):
                    status_message = node_output.get("status_message")

                    if status_message and (
                        not current_run_statuses
                        or status_message != current_run_statuses[-1]
                    ):
                        logger.info(f"[Graph Status] {status_message}")

                        new_status_message = status_message
                if new_status_message:
                    current_run_statuses.append(new_status_message)
                    status_markdown = "\n".join(
                        [f"- {s}" for s in current_run_statuses]
                    )
                    status_list_placeholder.markdown(status_markdown)

                if node_name in ["generate", "inform_no_rag_data"]:
                    logger.debug(f"Processing output from node '{node_name}'.")
                    generation_stream = (
                        node_output.get("generation_stream")
                        if isinstance(node_output, dict)
                        else None
                    )

                    if generation_stream and (
                        hasattr(generation_stream, "__iter__")
                        or hasattr(generation_stream, "__aiter__")
                    ):
                        try:
                            logger.debug(f"Iterating stream from '{node_name}'...")
                            chunk_count = 0
                            for chunk in generation_stream:
                                chunk_count += 1

                                chunk_content = (
                                    chunk.content
                                    if hasattr(chunk, "content")
                                    and isinstance(chunk.content, str)
                                    else chunk if isinstance(chunk, str) else None
                                )
                                if chunk_content:
                                    full_response += chunk_content
                                    response_placeholder.markdown(full_response + "▌")
                                elif chunk:
                                    logger.warning(
                                        f"Unexpected chunk type/content from '{node_name}': {type(chunk)}"
                                    )
                            logger.debug(
                                f"Finished stream from '{node_name}'. Chunks: {chunk_count}"
                            )
                            if chunk_count == 0:
                                logger.warning(
                                    f"Stream from '{node_name}' yielded no chunks."
                                )
                        except Exception as stream_ex:
                            logger.error(
                                f"Error iterating stream from '{node_name}': {stream_ex}",
                                exc_info=True,
                            )
                            graph_error = stream_ex

                            error_status_stream = (
                                f"❌ Error reading AI stream: {stream_ex}"
                            )
                            if (
                                not current_run_statuses
                                or error_status_stream != current_run_statuses[-1]
                            ):
                                current_run_statuses.append(error_status_stream)
                                status_list_placeholder.markdown(
                                    "\n".join([f"- {s}" for s in current_run_statuses])
                                )
                    elif generation_stream:
                        logger.error(
                            f"'generation_stream' from '{node_name}' not iterator: {type(generation_stream)}"
                        )
                    else:
                        logger.warning(
                            f"No valid 'generation_stream' in '{node_name}' output."
                        )

            logger.info("Finished processing graph stream events.")

            if graph_error:
                final_status_msg = f"❌ An error occurred: {graph_error}"

            else:
                final_status_msg = "✅ Completed!"

            if not full_response:
                if any("Missing data for RAG mode" in s for s in current_run_statuses):
                    logger.info("Completed with missing RAG data notification.")

                elif not graph_error:
                    logger.warning(
                        "No response content accumulated despite no apparent error."
                    )
                    full_response = "Sorry, I could not generate an answer."
                else:
                    full_response = f"Sorry, an error occurred: {graph_error}"

            if not current_run_statuses or final_status_msg != current_run_statuses[-1]:
                current_run_statuses.append(final_status_msg)
                status_list_placeholder.markdown(
                    "\n".join([f"- {s}" for s in current_run_statuses])
                )

            response_placeholder.markdown(full_response)

        except Exception as e:

            graph_error = e
            error_message = f"Critical error running graph: {graph_error}"
            logger.error(error_message, exc_info=True)
            final_status_msg = f"❌ {error_message}"

            if not current_run_statuses or final_status_msg != current_run_statuses[-1]:
                current_run_statuses.append(final_status_msg)
                status_list_placeholder.markdown(
                    "\n".join([f"- {s}" for s in current_run_statuses])
                )

            full_response = f"Sorry, a critical error occurred: {graph_error}"
            response_placeholder.markdown(full_response)

        logger.info(
            f"Final response generated (Mode: {st.session_state.mode_internal}, Length: {len(full_response)})."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
