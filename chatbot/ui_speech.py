import io
import logging

import speech_recognition as sr
import streamlit as st

logger = logging.getLogger("RAG_Chatbot_App")


def display_voice_recorder_st_native():
    audio_input_key = f"audio_input_{st.session_state.get('audio_input_version', 0)}"
    logger.debug(f"Rendering st.audio_input in sidebar with key: {audio_input_key}")

    audio_bytes_data = st.audio_input(label="Record Voice:", key=audio_input_key)

    recognized_text = None

    if audio_bytes_data:
        processed_audio_key = f"processed_audio_data_for_key_{audio_input_key}"

        if not st.session_state.get(processed_audio_key, False):
            st.session_state[processed_audio_key] = True

            wav_audio_data = audio_bytes_data.getvalue()

            try:
                r = sr.Recognizer()
                with sr.AudioFile(io.BytesIO(wav_audio_data)) as source:
                    audio_source_data = r.record(source)

                logger.info(
                    "Attempting to recognize speech using Google Web Speech API."
                )
                recognized_text = r.recognize_google(
                    audio_source_data, language="vi-VN"
                )

                if recognized_text:
                    logger.info(f"Speech recognized (from sidebar): {recognized_text}")

                else:
                    logger.warning(
                        "Google Web Speech API did not recognize any speech (from sidebar)."
                    )
                    st.warning("Không nhận dạng được giọng nói.", icon="🤔")

            except sr.UnknownValueError:
                logger.warning(
                    "Google Web Speech API could not understand the audio (from sidebar)."
                )
                st.warning("Không hiểu bạn nói gì.", icon="🤔")
            except sr.RequestError as e:
                logger.error(
                    f"Could not request results from Google Web Speech API (from sidebar); {e}"
                )
                st.error(f"Lỗi dịch vụ nhận dạng (sidebar).", icon="🔥")
            except Exception as e:
                logger.error(
                    f"Lỗi không mong muốn trong nhận dạng giọng nói (from sidebar): {e}",
                    exc_info=True,
                )
                st.error(f"Lỗi xử lý giọng nói (sidebar).", icon="🔥")
        else:
            logger.debug(
                f"Audio data for key {audio_input_key} already processed or awaiting next action."
            )

    if recognized_text:
        pass

    return recognized_text
