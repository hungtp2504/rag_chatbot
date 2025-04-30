import logging
import logging.handlers
import os
import sys
from dotenv import load_dotenv
from typing import Optional


class Settings:
    """
    Centralized configuration object holding constants and loaded environment variables.
    """

    def __init__(self):
        """Loads environment variables and sets configuration attributes."""
        logger = logging.getLogger("RAG_Chatbot_App")
        logger.info("Initializing application settings...")
        load_dotenv()

        self.OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME: Optional[str] = os.getenv("PINECONE_INDEX_NAME")
        self.TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")

        required_keys = {
            "OpenAI API Key": self.OPENAI_API_KEY,
            "Pinecone API Key": self.PINECONE_API_KEY,
            "Pinecone Index Name": self.PINECONE_INDEX_NAME,
            "Tavily API Key": self.TAVILY_API_KEY,
        }
        missing_keys = [name for name, value in required_keys.items() if not value]
        if missing_keys:
            error_msg = f"Error: Missing required environment variables: {', '.join(missing_keys)}"
            logger.critical(error_msg.replace("Error: ", ""))

            raise ValueError(error_msg)
        logger.info("Required environment variables loaded successfully.")

        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.TOP_K: int = int(os.getenv("TOP_K", "3"))
        self.TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "3"))

        self.PINECONE_BATCH_SIZE: int = int(os.getenv("PINECONE_BATCH_SIZE", "200"))
        self.PINECONE_MAX_REQUEST_BYTES: int = 2 * 1024 * 1024
        self.MAX_METADATA_FIELD_LENGTH: int = 1000
        self.MAX_TOTAL_METADATA_BYTES: int = 35 * 1024
        self.MAX_CHUNK_BYTES_WARNING: int = 1 * 1024 * 1024

        self.log_level_str: str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_level: int = getattr(logging, self.log_level_str, logging.INFO)
        verbose_str = os.getenv("APP_VERBOSE", "false").lower()
        self.app_verbose: bool = verbose_str in ["true", "1", "yes", "on"]
        self.app_log_file: str = "app.log"

        logger.info("Application settings initialized.")
        logger.debug(
            f"Settings loaded: ChunkSize={self.CHUNK_SIZE}, Overlap={self.CHUNK_OVERLAP}, BatchSize={self.PINECONE_BATCH_SIZE}, LogLevel={self.log_level_str}, Verbose={self.app_verbose}"
        )


try:
    settings = Settings()
except ValueError as e:

    logging.critical(f"Failed to initialize settings: {e}", exc_info=True)

    print(f"CRITICAL ERROR: Failed to initialize settings - {e}", file=sys.stderr)
    sys.exit(1)


LOG_FORMAT = (
    "%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging():
    """Configures logging for the entire application based on environment variables."""

    verbose_str = os.getenv("APP_VERBOSE", "false").lower()
    APP_VERBOSE = verbose_str in ["true", "1", "yes", "on"]
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file = settings.app_log_file if "settings" in globals() else "app.log"

    app_logger = logging.getLogger("RAG_Chatbot_App")
    app_logger.setLevel(log_level)
    app_logger.propagate = False

    if app_logger.hasHandlers():
        for handler in app_logger.handlers[:]:
            app_logger.removeHandler(handler)
            handler.close()

    if APP_VERBOSE:
        # print(f"--- CONFIG: VERBOSE MODE ENABLED: Logging to '{log_file}' at level '{logging.getLevelName(log_level)}' ---", file=sys.stderr)
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        try:
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            app_logger.addHandler(file_handler)

            app_logger.info(
                f"VERBOSE File Logging Started (Overwrite Mode). Level: {log_level_str}"
            )
        except Exception as e:
            # print(f"--- FATAL ERROR: Failed to configure file logging for {log_file}: {e} ---", file=sys.stderr)
            app_logger.addHandler(logging.NullHandler())
    else:

        app_logger.addHandler(logging.NullHandler())

    return app_logger
