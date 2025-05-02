import streamlit as st


def display_chat_history():
    """Displays the chat history from session state, including steps for AI messages."""
    for message in st.session_state.get("messages", []):
        role = message.get("role", "assistant")
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):

            st.markdown(message.get("content", ""))

            if role == "assistant" and "steps" in message and message["steps"]:

                with st.expander("⚙️ Detailed Processing Steps:", expanded=False):

                    status_markdown = "\n".join([f"- {s}" for s in message["steps"]])
                    st.markdown(status_markdown)


def display_ai_message_elements():
    """Creates and returns placeholders for the *current* AI response and status list."""

    response_placeholder = st.empty()

    with st.expander("⚙️ Detailed Processing Steps (Current Run):", expanded=True):
        status_list_placeholder = st.empty()

    return response_placeholder, status_list_placeholder
