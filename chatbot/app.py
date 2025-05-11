import logging
from typing import Any, Dict, Generator, List

import streamlit as st

st.set_page_config(page_title="Versatile Chatbot", layout="wide")

try:
    from chatbot.config import settings, setup_logging
    from chatbot.processing import process_rag_sources
    from chatbot.state_manager import initialize_session_state
    from chatbot.ui_chat import display_ai_message_elements, display_chat_history
    from chatbot.ui_sidebar import display_sidebar
    from llm.graph_logic import GraphState
except ImportError as e:
    try:
        from config import settings, setup_logging
        from processing import process_rag_sources
        from state_manager import initialize_session_state
        from ui_chat import display_ai_message_elements, display_chat_history
        from ui_sidebar import display_sidebar

        from llm.graph_logic import GraphState

        st.warning(
            "Using fallback imports. This might not work correctly if the project structure is complex or if files are not in the expected locations."
        )
        if "streamlit_defined_logger" not in st.session_state:
            st.session_state.streamlit_defined_logger = True
    except ImportError as e_fallback:
        st.error(
            f"Could not import necessary project modules: {e}. Fallback error: {e_fallback}. Please check your paths and directory structure."
        )
        st.stop()

logger = setup_logging()
initialize_session_state()

st.title("Versatile Chatbot (RAG / Web Search / Voice)")

css = r"""
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
"""
st.markdown(css, unsafe_allow_html=True)


uploaded_files, urls, process_rag_button_clicked, transcribed_prompt_from_sidebar = (
    display_sidebar()
)

if process_rag_button_clicked:
    if uploaded_files or (urls and urls.strip()):
        process_rag_sources(uploaded_files, urls)
    else:
        logger.warning(
            "RAG processing button was clicked, but no RAG sources (files or URLs) were provided."
        )
        st.warning("Please provide PDF files or enter URLs to process for RAG mode.")

current_mode_internal = st.session_state.get("mode_internal", "Web Search")
display_mode_text = (
    "Web Search (Tavily)"
    if current_mode_internal == "Web Search"
    else "RAG (Uploaded Documents)"
)
st.markdown(f"**Current operating mode:** {display_mode_text}")


display_chat_history()


typed_prompt = st.chat_input("Nhập câu hỏi của bạn ở đây...")

current_prompt_to_process = None
if transcribed_prompt_from_sidebar:
    current_prompt_to_process = transcribed_prompt_from_sidebar
    logger.info(f"Voice input from sidebar: '{current_prompt_to_process}'")
elif typed_prompt:
    current_prompt_to_process = typed_prompt
    logger.info(f"Text input from chat_input: '{current_prompt_to_process}'")


if current_prompt_to_process:

    logger.info(
        f"Processing question (Mode: {current_mode_internal}, Length: {len(current_prompt_to_process)})."
    )
    st.session_state.messages.append(
        {"role": "user", "content": current_prompt_to_process}
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(current_prompt_to_process)

    with st.chat_message("ai", avatar="🤖"):
        response_placeholder, status_list_placeholder = display_ai_message_elements()

        if "graph" not in st.session_state or st.session_state.graph is None:
            error_content = "Error: Chat logic graph has not been initialized. Please try refreshing the page."

            response_placeholder.markdown(error_content)
            status_list_placeholder.markdown("- ❌ Graph Initialization Error")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_content,
                    "steps": ["❌ Graph Initialization Error"],
                }
            )
        else:

            compiled_graph = st.session_state.graph
            full_response = ""
            graph_error_occurred = None
            history_for_graph = st.session_state.messages[:-1]

            final_run_status_history: List[str] = [
                "▶️ Starting to process the question..."
            ]
            if status_list_placeholder:
                status_list_placeholder.markdown("- " + final_run_status_history[-1])

            try:
                graph_input: GraphState = {
                    "query": current_prompt_to_process,
                    "chat_history": history_for_graph,
                    "chatbot_mode": current_mode_internal,
                    "use_rag": st.session_state.get("vectorstore") is not None,
                    "context": None,
                    "generation": None,
                    "generation_stream": None,
                    "status_message": "Starting processing...",
                    "web_search_sources": None,
                    "rag_chunk_ids": None,
                    "rag_chunk_scores": None,
                    "run_status_history": list(final_run_status_history),
                }

                events: Generator[Dict[str, Any], None, None] = compiled_graph.stream(
                    graph_input, config={"recursion_limit": 10}
                )

                for event in events:
                    logger.debug(f"Event from Graph: {event}")
                    event_keys = list(event.keys())
                    if not event_keys or event_keys[0] == "__end__":
                        if event_keys and event_keys[0] == "__end__":
                            final_node_output = event.get(event_keys[0])
                            if isinstance(
                                final_node_output, dict
                            ) and final_node_output.get("run_status_history"):
                                final_run_status_history = final_node_output.get(
                                    "run_status_history", final_run_status_history
                                )
                        continue

                    node_name = event_keys[0]
                    node_output = event.get(node_name)

                    if isinstance(node_output, dict):
                        new_history_from_node = node_output.get("run_status_history")
                        if new_history_from_node and isinstance(
                            new_history_from_node, list
                        ):
                            final_run_status_history = new_history_from_node

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
                                            if response_placeholder:
                                                response_placeholder.markdown(
                                                    full_response + "▌"
                                                )
                                except Exception as stream_ex:
                                    logger.error(
                                        f"Error iterating through stream from node '{node_name}': {stream_ex}",
                                        exc_info=True,
                                    )
                                    graph_error_occurred = stream_ex
                                    final_run_status_history.append(
                                        f"⚠️ Stream Error: {stream_ex}"
                                    )
                                    break

                    if status_list_placeholder:
                        status_markdown = "\n".join(
                            [f"- {s}" for s in final_run_status_history]
                        )
                        status_list_placeholder.markdown(status_markdown)

                    if graph_error_occurred:
                        break

                if graph_error_occurred:
                    error_message_for_user = f"Sorry, an error occurred during processing: {graph_error_occurred}"
                    if not full_response:
                        full_response = error_message_for_user
                    if (
                        final_run_status_history
                        and final_run_status_history[-1]
                        != f"❌ Critical Error: {graph_error_occurred}"
                    ):
                        final_run_status_history.append(
                            f"❌ Error: {graph_error_occurred}"
                        )
                else:
                    if not full_response:
                        full_response = "Sorry, I couldn't find the information or generate an answer."
                        final_run_status_history.append("⚠️ No response generated.")
                    else:
                        final_run_status_history.append("✅ Completed!")

                if status_list_placeholder:
                    status_markdown_final = "\n".join(
                        [f"- {s}" for s in final_run_status_history]
                    )
                    status_list_placeholder.markdown(status_markdown_final)
                if response_placeholder:
                    response_placeholder.markdown(full_response)

            except Exception as e:

                logger.error(
                    f"Critical error during chat logic execution (graph execution): {e}",
                    exc_info=True,
                )
                graph_error_occurred = e
                full_response = f"Sorry, a critical system error occurred: {e}"
                final_run_status_history.append(f"❌ System Error: {e}")

                if response_placeholder:
                    response_placeholder.markdown(full_response)
                if status_list_placeholder:
                    status_markdown_exc = "\n".join(
                        [f"- {s}" for s in final_run_status_history]
                    )
                    status_list_placeholder.markdown(status_markdown_exc)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "steps": final_run_status_history,
                }
            )

            if (
                transcribed_prompt_from_sidebar
                and current_prompt_to_process == transcribed_prompt_from_sidebar
            ):
                if "audio_input_version" in st.session_state:
                    st.session_state.audio_input_version += 1
                    logger.debug(
                        f"Incremented audio_input_version to: {st.session_state.audio_input_version} after processing sidebar voice input."
                    )
