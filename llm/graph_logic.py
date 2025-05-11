import logging
from typing import Any, Dict, Generator, List, Literal, Optional, TypedDict

import streamlit as st
from langgraph.graph import END, StateGraph

try:
    from .components import (
        create_generation_chain,
        create_rag_prompt,
        create_web_search_prompt,
        format_chat_history,
        get_llm,
        similarity_search,
        web_search,
    )
except ImportError:

    try:
        from llm.components import (
            create_generation_chain,
            create_rag_prompt,
            create_web_search_prompt,
            format_chat_history,
            get_llm,
            similarity_search,
            web_search,
        )
    except ImportError as e:
        logging.error(
            f"Failed to import components for graph_logic: {e}", exc_info=True
        )

        try:
            st.error(
                f"Critical Import Error: Could not load components for graph logic. Details: {e}"
            )
            st.stop()
        except Exception:
            raise ImportError(
                f"Could not import components needed by graph_logic: {e}"
            ) from e


logger = logging.getLogger("RAG_Chatbot_App")


class GraphState(TypedDict):
    """Defines the state passed between nodes in the graph."""

    query: str
    chat_history: List[dict]
    chatbot_mode: str
    use_rag: bool
    context: Optional[str]
    generation: Optional[str]
    generation_stream: Optional[Generator[Any, None, None]]
    status_message: str
    web_search_sources: Optional[List[str]]
    rag_chunk_ids: Optional[List[str]]
    rag_chunk_scores: Optional[List[float]]
    run_status_history: List[str]


def route_query(
    state: GraphState,
) -> Literal["vectorstore", "web_search"]:
    """Routes the query based ONLY on the selected chatbot mode."""
    mode = state.get("chatbot_mode")
    vectorstore_ready = state.get("use_rag", False)
    logger.info(
        f"Routing query - Mode Selected: {mode}, Vectorstore Initialized: {vectorstore_ready}"
    )

    if mode == "RAG":

        if not vectorstore_ready:

            logger.warning(
                "RAG mode selected, but Pinecone vectorstore is not initialized in session state. Proceeding to retrieve_context anyway."
            )
        logger.info("RAG Mode selected -> Route: vectorstore")
        return "vectorstore"
    elif mode == "Web Search":
        logger.info("Web Search Mode selected -> Route: web_search")
        return "web_search"
    else:

        logger.warning(
            f"Unknown chatbot mode: '{mode}'. Defaulting route to 'web_search'."
        )
        return "web_search"


def inform_no_rag_data(state: GraphState) -> Dict[str, Any]:
    """Generates a notification stream when RAG mode lacks data."""
    logger.info("Executing 'inform_no_rag_data' node.")
    message = "You selected RAG mode, but no data sources have been processed yet. Please upload and process documents first, or switch to 'Web Search (Tavily)' mode."
    status = "Missing data for RAG mode"

    def single_message_stream(msg: str):
        yield msg

    return {
        "generation_stream": single_message_stream(message),
        "status_message": status,
        "context": "",
    }


def retrieve_context(state: GraphState) -> Dict[str, Any]:
    """Retrieves context, IDs, and scores from the vector store for RAG."""
    logger.info("Executing 'retrieve_context' node (RAG).")
    query = state["query"]
    vectorstore = st.session_state.get("vectorstore", None)
    status_update = "Querying RAG Vector Store..."

    run_history = state.get("run_status_history", [])
    run_history.append(status_update)

    context = ""
    chunk_ids = []
    chunk_scores = []

    if vectorstore:
        try:
            context, chunk_ids, chunk_scores = similarity_search(query, vectorstore)
            if context:
                status_update = f"Retrieved context from {len(chunk_ids)} RAG chunks."

                if chunk_ids and chunk_scores:
                    chunk_details = [
                        f"  - Chunk {i+1}: ID=`{chunk_id}` (Score: {score:.4f})"
                        for i, (chunk_id, score) in enumerate(
                            zip(chunk_ids, chunk_scores)
                        )
                    ]
                    detail_status = "Retrieved RAG Chunks:\n" + "\n".join(chunk_details)
                    run_history.append(detail_status)
                else:
                    run_history.append(status_update)
            else:
                status_update = "No relevant RAG context found."
                run_history.append(status_update)
        except Exception as e:
            logger.error(f"Error during RAG similarity_search call: {e}", exc_info=True)
            status_update = f"Error querying RAG Vector Store: {e}"
            run_history.append(status_update)
            context, chunk_ids, chunk_scores = "", [], []
    else:
        status_update = "Error: RAG Vector Store not initialized or available."
        run_history.append(status_update)

    return {
        "context": context or "",
        "status_message": status_update,
        "rag_chunk_ids": chunk_ids,
        "rag_chunk_scores": chunk_scores,
        "run_status_history": run_history,
    }


def search_web(state: GraphState) -> Dict[str, Any]:
    """Performs web search using Tavily and stores context and sources."""
    logger.info("Executing 'search_web' node.")
    query = state["query"]
    status_update = "Searching the web..."
    run_history = state.get("run_status_history", [])
    run_history.append(status_update)

    context = ""
    source_urls = []

    try:
        context, source_urls = web_search(query)
        if context:
            status_update = (
                f"Retrieved information from {len(source_urls)} web sources."
            )

            if source_urls:
                source_list_str = "\n".join([f"  - `{s}`" for s in source_urls])
                detail_status = f"Retrieved Web Sources:\n{source_list_str}"
                run_history.append(detail_status)
            else:
                run_history.append(status_update)
        else:
            status_update = "No relevant information found on the web."
            run_history.append(status_update)
    except Exception as e:
        logger.error(f"Error during web search execution: {e}", exc_info=True)
        status_update = f"Error searching web: {e}"
        run_history.append(status_update)
        context, source_urls = "", []

    return {
        "context": context or "",
        "status_message": status_update,
        "web_search_sources": source_urls,
        "run_status_history": run_history,
    }


def generate_answer(state: GraphState) -> Dict[str, Any]:
    logger.info("Executing 'generate_answer' node.")
    query = state["query"]
    context = state.get("context", "") # Lấy context, mặc định là chuỗi rỗng nếu không có
    chat_history_list = state.get("chat_history", [])
    run_history = state.get("run_status_history", []) # Lấy lịch sử chạy hiện tại
    mode = state.get("chatbot_mode")

    # 1. Lấy LLM
    llm = get_llm()
    if not llm:
        logger.error("LLM not available in generate_answer node.")
        def error_stream_no_llm():
            yield "Error: Language Model is unavailable."
        run_history.append("❌ LLM Initialization Error: Model unavailable.")
        return {
            "generation_stream": error_stream_no_llm(),
            "status_message": "LLM Initialization Error",
            "run_status_history": run_history,
        }

    # 2. Định dạng lịch sử chat
    formatted_history = format_chat_history(chat_history_list)

    # 3. Chọn prompt template dựa trên chế độ
    prompt_template = None
    if mode == "RAG":
        logger.debug("Using RAG prompt for generation.")
        prompt_template = create_rag_prompt()
    elif mode == "Web Search":
        logger.debug("Using Web Search prompt for generation.")
        prompt_template = create_web_search_prompt()
    else:
        logger.warning(
            f"Unknown mode '{mode}' in generate node, using Web Search prompt."
        )
        prompt_template = create_web_search_prompt()
        run_history.append(f"⚠️ Unknown mode '{mode}', defaulted to Web Search prompt.")

    # 4. Tạo generation chain (chỉ một lần)
    generation_chain = None
    try:
        generation_chain = create_generation_chain(llm, prompt_template)
        run_history.append("✅ Generation chain created.")
        logger.info("Generation chain created successfully.")
    except ValueError as e:
        logger.error(f"Failed to create generation chain: {e}", exc_info=True)
        def error_stream_chain_creation():
            yield f"Error: Could not create generation chain - {e}"
        run_history.append(f"❌ Chain Creation Error: {e}")
        return {
            "generation_stream": error_stream_chain_creation(),
            "status_message": "Chain Creation Error",
            "run_status_history": run_history,
        }

    # 5. Chuẩn bị đầu vào cho chain
    chain_input = {
        "query": query,
        "context": context if context else "No context provided.", # Đảm bảo có giá trị mặc định
        "chat_history": formatted_history,
    }

    stream_to_return = iter([])  # Mặc định là một iterator rỗng
    final_status = "🚀 Starting answer generation..."
    run_history.append(final_status)

    # 6. Thực thi chain và stream kết quả
    try:
        logger.info(f"Calling stream on generation_chain with input: query='{str(query)[:50]}...', context_length={len(context)}")
        _raw_stream = generation_chain.stream(chain_input)

        if hasattr(_raw_stream, "__iter__") or hasattr(_raw_stream, "__aiter__"):
            stream_to_return = _raw_stream
            final_status = "📥 Receiving data from LLM..."
            logger.info("Successfully started streaming from LLM.")
        else:
            logger.error(
                f"generation_chain.stream did not return an iterator! Type: {type(_raw_stream)}"
            )
            final_status = "Error: Did not receive a valid stream from LLM."
            def error_stream_no_iterator():
                yield "Error: LLM did not return a valid stream object."
            stream_to_return = error_stream_no_iterator()
        run_history.append(final_status) # Cập nhật trạng thái sau khi thử stream

    except Exception as e:
        logger.error(f"Error calling generation_chain.stream: {e}", exc_info=True)
        final_status = f"Error generating answer: {e}"
        run_history.append(f"❌ Error during answer generation: {e}")
        def error_stream_exception_in_stream():
            yield f"Sorry, an error occurred while generating the answer: {e}"
        stream_to_return = error_stream_exception_in_stream()

    return {
        "generation_stream": stream_to_return,
        "status_message": final_status,
        "run_status_history": run_history,
    }


def build_graph() -> StateGraph:
    logger.info("Building the application graph...")
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_context)
    graph.add_node("web_search", search_web)
    graph.add_node("generate", generate_answer)

    logger.debug(
        "Adding conditional edges from entry point (__start__) based on route_query."
    )
    graph.add_conditional_edges(
        "__start__",
        route_query,
        {
            "vectorstore": "retrieve",
            "web_search": "web_search",
        },
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("web_search", "generate")

    graph.add_edge("generate", END)

    try:
        app_graph = graph.compile()
        logger.info("Graph compiled successfully with updated RAG routing.")
        return app_graph
    except Exception as e:
        logger.error(f"Failed to compile the graph: {e}", exc_info=True)
        try:
            st.error(
                f"Critical Error: Failed to compile application graph. Details: {e}"
            )
            st.stop()
        except Exception:
            raise RuntimeError(f"Failed to compile application graph: {e}") from e
