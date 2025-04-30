import logging
import logging.handlers
import os
import subprocess
import sys

try:
    from chatbot.config import setup_logging

    logger = setup_logging()

except ImportError:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.warning(
        "main.py: Could not load logging configuration from app.config. Using basic config."
    )
except Exception as e:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.error(
        f"main.py: Error configuring logging from app.config: {e}", exc_info=True
    )


def run_streamlit():
    """Launches the Streamlit application app/app.py using subprocess."""

    app_file = os.path.join("chatbot", "app.py")
    logger.info(f"main.py: Looking for application file: {app_file}")

    if not os.path.exists(app_file):
        logger.error(
            f"main.py: Error - Could not find file '{app_file}'. Make sure it exists relative to main.py."
        )
        print(
            f"Error: Cannot find {app_file}. Please ensure it's in the 'app' subdirectory.",
            file=sys.stderr,
        )
        sys.exit(1)

    command = [sys.executable, "-m", "streamlit", "run", app_file]
    logger.info(f"main.py: Preparing to execute command: {' '.join(command)}")
    print(f"--- Starting Streamlit app ({app_file}) ---")

    try:

        process = subprocess.run(command, check=True)

        logger.info(
            f"main.py: Streamlit process finished with return code: {process.returncode}"
        )

    except subprocess.CalledProcessError as e:

        logger.error(
            f"main.py: Error running Streamlit application: {e}", exc_info=True
        )
        print(f"\n--- Error running Streamlit app ---", file=sys.stderr)
        print(f"Command: {' '.join(command)}", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

    except FileNotFoundError:

        logger.error(
            "main.py: Error - 'streamlit' command or Python executable not found.",
            exc_info=True,
        )
        print(f"\n--- Error: Could not execute Streamlit command ---", file=sys.stderr)
        print(
            f"Ensure Streamlit is installed in the Python environment ('{sys.executable}')",
            file=sys.stderr,
        )
        print(f"and that the Python executable path is correct.", file=sys.stderr)
        print(
            f"You can install Streamlit using: pip install streamlit", file=sys.stderr
        )
        sys.exit(1)

    except KeyboardInterrupt:

        logger.info(
            "main.py: Streamlit application stopped by user (KeyboardInterrupt)."
        )
        print("\n--- Streamlit app interrupted by user ---")
        sys.exit(0)


if __name__ == "__main__":
    run_streamlit()
