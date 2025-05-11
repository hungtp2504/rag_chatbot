import logging
import uuid

import streamlit as st

try:
    from .ui_speech import display_voice_recorder_st_native
except ImportError:
    from chatbot.ui_speech import display_voice_recorder_st_native


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


def _make_actual_api_call(note_title: str, note_body: str) -> dict:
    logger.info(f"Simulating API call to create note: Title='{note_title}'")
    if "fail" in note_title.lower():
        return {"success": False, "error": "Simulated tool failure."}
    else:
        return {"success": True, "note_id": str(uuid.uuid4())}


def call_create_note_tool(title: str, text_content: str) -> bool:
    logger.debug(f"Attempting to call note creation tool with Title: '{title}'")
    try:
        response = _make_actual_api_call(note_title=title, note_body=text_content)
        if response and isinstance(response, dict) and response.get("success"):
            note_id = response.get("note_id", "[N/A]")
            logger.info(f"Successfully created note via tool. Note ID: {note_id}")
            return True
        else:
            error_msg = (
                response.get("error", "Unknown tool error")
                if isinstance(response, dict)
                else "Invalid response from tool"
            )
            logger.error(f"Note creation tool failed: {error_msg}")
            return False
    except Exception as e:
        logger.error(
            f"Exception occurred while calling create_note tool function: {e}",
            exc_info=True,
        )
        return False


def handle_note_creation():
    logger.info("Executing handle_note_creation callback for session notes.")
    title_key = "note_title_sidebar"
    content_key = "note_content_sidebar"

    title = st.session_state.get(title_key, "").strip()
    content = st.session_state.get(content_key, "").strip()

    if not title and not content:
        st.toast("Please enter a title or content.", icon="⚠️")
        logger.warning("Note creation callback skipped: No title or content.")
        return

    final_title = title if title else ("Untitled Note" if content else None)

    if final_title or content:
        try:
            new_note = {
                "id": str(uuid.uuid4())[:8],
                "title": final_title,
                "content": content,
            }

            if "session_notes" not in st.session_state:
                st.session_state.session_notes = []
            st.session_state.session_notes.append(new_note)
            logger.info(f"Note '{final_title}' added to session state.")

            st.session_state[title_key] = ""
            st.session_state[content_key] = ""
            st.toast("Note added to session!", icon="✅")

        except Exception as e:
            st.toast(f"Error adding note: {e}", icon="🔥")
            logger.error(f"Error adding note to session state: {e}", exc_info=True)


def handle_note_deletion(note_index_to_delete: int):
    """Callback to remove a note from the session state list by its index."""
    logger.info(
        f"Executing handle_note_deletion callback for index: {note_index_to_delete}"
    )
    try:
        if "session_notes" in st.session_state and st.session_state.session_notes:
            if 0 <= note_index_to_delete < len(st.session_state.session_notes):
                deleted_note = st.session_state.session_notes.pop(note_index_to_delete)
                logger.info(
                    f"Deleted note '{deleted_note.get('title')}' at index {note_index_to_delete}."
                )
                st.toast("Note deleted!", icon="🗑️")
            else:
                logger.warning(
                    f"Invalid index provided for deletion: {note_index_to_delete}"
                )
                st.toast("Error: Could not delete note (invalid index).", icon="🔥")
        else:
            logger.warning(
                "Attempted to delete note, but session_notes list is empty or missing."
            )
            st.toast("Error: No notes found to delete.", icon="🔥")
    except Exception as e:
        logger.error(
            f"Error deleting note at index {note_index_to_delete}: {e}", exc_info=True
        )
        st.toast(f"Error deleting note: {e}", icon="🔥")


def display_sidebar():
    """Displays all sidebar content, including voice recorder, RAG, and Notes."""
    transcribed_prompt_from_sidebar = None
    uploaded_files_sidebar = None
    urls_sidebar = ""
    process_rag_button_sidebar = False

    with st.sidebar:
        st.header("Chatbot Configuration")

        mode_options = ["Web Search (Tavily)", "RAG (Uploaded Documents)"]

        current_mode_internal = st.session_state.get("mode_internal", "Web Search")

        if current_mode_internal == "RAG":
            initial_radio_index = 1
        else:
            initial_radio_index = 0

        st.radio(
            "Select operating mode:",
            options=mode_options,
            key="chatbot_mode_selector",
            index=initial_radio_index,
            on_change=update_chatbot_mode,
        )
        st.markdown("---")

        st.subheader("🎙️ Recording")
        transcribed_prompt_from_sidebar = display_voice_recorder_st_native()
        st.markdown("---")

        if st.session_state.get("mode_internal") == "RAG":
            with st.container():
                st.header("RAG Data Sources")
                uploaded_files_sidebar = st.file_uploader(
                    "Upload PDF files",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key="pdf_uploader_sidebar",
                )
                urls_sidebar = st.text_area(
                    "Or enter URLs (one per line)", key="url_input_sidebar"
                )
                process_rag_button_sidebar = st.button(
                    "Process RAG Sources", key="process_sources_btn_sidebar"
                )

                if st.session_state.get("processed_sources"):
                    st.subheader("Processed Sources (for RAG):")
                    for src in st.session_state.processed_sources:
                        st.write(f"- {src}")
            st.markdown("---")

        st.header("Notes")
        with st.expander("Add a New Note", expanded=False):
            title_key = "note_title_sidebar"
            content_key = "note_content_sidebar"
            with st.form(key="note_form_sidebar_v2_unique"):
                st.text_input(
                    "Note Title:", key=title_key, placeholder="E.g., Meeting Summary"
                )
                st.text_area(
                    "Note Content:",
                    key=content_key,
                    placeholder="Type your note here...",
                    height=120,
                )
                st.form_submit_button(
                    "Add Note to Session",
                    on_click=handle_note_creation,
                    use_container_width=True,
                )

        if "session_notes" in st.session_state and st.session_state.session_notes:
            st.subheader("Existing Notes")
            notes_list = list(st.session_state.session_notes)
            for i, note in enumerate(notes_list):
                note_title = note.get("title", "Untitled")
                note_content = note.get("content", "*No content*")
                note_id = note.get("id", f"note_default_id_{i}")

                col_expander, col_button = st.columns([0.85, 0.15])
                with col_expander:
                    with st.expander(f"{note_title}", expanded=False):
                        st.markdown(note_content)
                with col_button:
                    delete_key = f"delete_note_{note_id}_{i}_sidebar"
                    st.button(
                        "🗑️",
                        type="tertiary",
                        use_container_width=True,
                        key=delete_key,
                        on_click=handle_note_deletion,
                        args=(i,),
                        help="Delete this note",
                    )
            st.markdown("---")

    return (
        uploaded_files_sidebar,
        urls_sidebar,
        process_rag_button_sidebar,
        transcribed_prompt_from_sidebar,
    )
