import streamlit as st


def display_chat_history():
    """Displays the chat history from session state."""

    for message in st.session_state.get("messages", []):

        avatar = "👤" if message.get("role") == "user" else "🤖"

        with st.chat_message(message.get("role", "assistant"), avatar=avatar):

            st.markdown(message.get("content", ""))


def display_ai_message_elements():
    """Creates and returns placeholders for the AI response and status list."""
    response_placeholder = st.empty()

    with st.expander("⚙️ Detailed Processing Steps:", expanded=False):
        status_list_placeholder = st.empty()

    return response_placeholder, None, status_list_placeholder
