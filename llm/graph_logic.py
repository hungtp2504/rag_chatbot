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
        from components import (
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
    """Retrieves context from the vector store for RAG."""
    logger.info("Executing 'retrieve_context' node (RAG).")
    query = state["query"]
    vectorstore = st.session_state.get("vectorstore", None)
    status_update = "Querying RAG Vector Store..."
    context = ""

    if vectorstore:
        try:
            context = similarity_search(query, vectorstore)
            if context:
                status_update = "Retrieved context from RAG Vector Store."
                logger.info(
                    f"RAG context retrieval successful (length {len(context)})."
                )
            else:
                status_update = "No relevant RAG context found."
                logger.info("No relevant RAG context found.")
        except Exception as e:
            logger.error(f"Error during RAG query: {e}", exc_info=True)
            status_update = f"Error querying RAG: {e}"
            context = ""
    else:
        logger.error("RAG Vector Store not available in retrieve_context node.")
        status_update = "Error: RAG Vector Store not ready."

    return {"context": context or "", "status_message": status_update}


def search_web(state: GraphState) -> Dict[str, Any]:
    """Performs web search using Tavily."""
    logger.info("Executing 'search_web' node.")
    query = state["query"]
    status_update = "Searching the web..."
    context = ""
    try:
        context = web_search(query)
        if context:
            status_update = "Retrieved information from web search."
            logger.info(f"Web search successful (length {len(context)}).")
        else:
            status_update = "No relevant information found on the web."
            logger.info("Web search returned no results.")
    except Exception as e:
        logger.error(f"Error during web search: {e}", exc_info=True)
        status_update = f"Error searching web: {e}"
        context = ""

    return {"context": context or "", "status_message": status_update}


def generate_answer(state: GraphState) -> Dict[str, Any]:
    """Generates the final answer stream based on context and mode."""
    logger.info("Executing 'generate_answer' node.")
    query = state["query"]
    context = state.get("context", "")
    chat_history_list = state.get("chat_history", [])
    mode = state.get("chatbot_mode")

    llm = get_llm()
    if not llm:
        logger.error("LLM not available in generate_answer node.")

        def error_stream():
            yield "Error: Language Model is unavailable."

        return {
            "generation_stream": error_stream(),
            "status_message": "LLM Initialization Error",
        }

    formatted_history = format_chat_history(chat_history_list)

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

    try:
        generation_chain = create_generation_chain(llm, prompt_template)
    except ValueError as e:
        logger.error(f"Failed to create generation chain: {e}", exc_info=True)

        def error_stream():
            yield f"Error: Could not create generation chain - {e}"

        return {
            "generation_stream": error_stream(),
            "status_message": "Chain Creation Error",
        }

    chain_input = {
        "query": query,
        "context": context if context else "No context provided.",
        "chat_history": formatted_history,
    }
    logger.debug(f"Input context length for LLM: {len(chain_input['context'])}")

    stream_to_return = iter([])
    final_status = "Starting answer generation..."
    try:
        logger.info("Calling stream on generation_chain...")
        _raw_stream = generation_chain.stream(chain_input)
        if hasattr(_raw_stream, "__iter__") or hasattr(_raw_stream, "__aiter__"):
            stream_to_return = _raw_stream
            final_status = "Receiving data from LLM..."
            logger.info("Successfully started streaming from LLM.")
        else:
            logger.error(
                f"generation_chain.stream did not return an iterator! Type: {type(_raw_stream)}"
            )
            final_status = "Error: Did not receive a valid stream from LLM."
    except Exception as e:
        logger.error(f"Error calling generation_chain.stream: {e}", exc_info=True)
        final_status = f"Error generating answer: {e}"

    return {
        "generation_stream": stream_to_return,
        "status_message": final_status,
    }


def build_graph() -> StateGraph:
    logger.info("Building the application graph...")
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_context)
    graph.add_node("web_search", search_web)
    graph.add_node("generate", generate_answer)
    graph.add_node("inform_no_rag_data", inform_no_rag_data)

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
