import os
import fnmatch
import logging
from typing import List, Optional

# Configure basic logging for errors during file processing
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


def write_project_to_txt(
    project_dir: str,
    output_file: str,
    exclude_patterns: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None,
    max_file_size_kb: Optional[int] = None,
    log_skipped: bool = True,
):
    if exclude_patterns is None:
        # Default common exclusions
        exclude_patterns = [
            ".git*",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.log",
            ".env",
            "venv",
            ".venv",
            "node_modules",
            "dist",
            "build",
            "*.egg-info",
            ".DS_Store",
            ".idea",
            ".vscode",
            os.path.basename(output_file),
        ]
        logging.info(f"Using default exclude patterns: {exclude_patterns}")

    # Get the absolute path of the output file to ensure accurate exclusion
    abs_output_file = os.path.abspath(output_file)

    if not os.path.isdir(project_dir):
        logging.error(
            f"Error: Project directory '{project_dir}' not found or is not a directory."
        )
        return

    try:
        with open(output_file, "w", encoding="utf-8", errors="ignore") as outfile:
            # Write header
            outfile.write(
                f"# Project Structure and Content for: {os.path.abspath(project_dir)}\n"
            )
            outfile.write(f"# Output generated to: {abs_output_file}\n")
            outfile.write("=" * 80 + "\n\n")

            for root, dirs, files in os.walk(project_dir, topdown=True):
                # --- Exclusion/Inclusion Logic ---
                # Filter directories based on patterns
                original_dirs = dirs[:]  # Copy list before modifying
                dirs[:] = [
                    d
                    for d in dirs
                    if not any(
                        fnmatch.fnmatch(d, pattern) for pattern in exclude_patterns
                    )
                    and (
                        include_patterns is None
                        or any(
                            fnmatch.fnmatch(os.path.join(root, d), pattern)
                            for pattern in include_patterns
                        )
                        or any(
                            fnmatch.fnmatch(d, pattern) for pattern in include_patterns
                        )
                    )  # Check dir name too
                ]
                # Log skipped directories
                if log_skipped:
                    skipped_dirs = set(original_dirs) - set(dirs)
                    for skipped_d in skipped_dirs:
                        logging.info(
                            f"Skipping directory: {os.path.join(root, skipped_d)}"
                        )

                # Calculate depth for indentation
                relative_path = os.path.relpath(root, project_dir)
                depth = relative_path.count(os.sep)
                indent = "    " * depth  # 4 spaces per level

                # Write current directory path
                if relative_path == ".":
                    outfile.write(f"[{os.path.basename(project_dir)}/]\n")
                else:
                    outfile.write(f"{indent}[{os.path.basename(root)}/]\n")

                # Process files in the current directory
                sub_indent = "    " * (depth + 1)
                for filename in sorted(files):
                    file_path = os.path.join(root, filename)
                    abs_file_path = os.path.abspath(file_path)

                    # Check exclusion patterns for files
                    if any(
                        fnmatch.fnmatch(filename, pattern)
                        for pattern in exclude_patterns
                    ) or any(
                        fnmatch.fnmatch(file_path, pattern)
                        for pattern in exclude_patterns
                    ):
                        if log_skipped:
                            logging.info(f"Skipping excluded file: {file_path}")
                        continue

                    # Check inclusion patterns for files (if provided)
                    if (
                        include_patterns
                        and not any(
                            fnmatch.fnmatch(filename, pattern)
                            for pattern in include_patterns
                        )
                        and not any(
                            fnmatch.fnmatch(file_path, pattern)
                            for pattern in include_patterns
                        )
                    ):
                        if log_skipped:
                            logging.info(
                                f"Skipping file not in include list: {file_path}"
                            )
                        continue

                    # Ensure we don't include the output file itself
                    if abs_file_path == abs_output_file:
                        if log_skipped:
                            logging.info(f"Skipping output file itself: {file_path}")
                        continue

                    # Write filename
                    outfile.write(f"{sub_indent}{filename}\n")

                    # --- Read and Write File Content ---
                    try:
                        # Check file size limit
                        if max_file_size_kb is not None:
                            file_size = os.path.getsize(file_path)
                            if file_size > max_file_size_kb * 1024:
                                outfile.write(
                                    f"{sub_indent}    [CONTENT SKIPPED - File size ({file_size / 1024:.2f} KB) > limit ({max_file_size_kb} KB)]\n\n"
                                )
                                if log_skipped:
                                    logging.info(
                                        f"Skipping content (size limit): {file_path}"
                                    )
                                continue

                        # Try reading with UTF-8 first, fallback if needed (already handled by errors='ignore' in open)
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as infile:
                            content = infile.read()

                        # Write content delimiter and content
                        outfile.write(f"{sub_indent}    --- START FILE CONTENT ---\n")
                        # Indent each line of the content
                        for line in content.splitlines():
                            outfile.write(f"{sub_indent}    {line}\n")
                        outfile.write(f"{sub_indent}    --- END FILE CONTENT ---\n\n")

                    except Exception as e:
                        outfile.write(f"{sub_indent}    [ERROR READING FILE: {e}]\n\n")
                        logging.error(
                            f"Error reading file {file_path}: {e}", exc_info=False
                        )  # Log error without full traceback usually

            outfile.write("=" * 80 + "\n")
            outfile.write("# End of Project Structure and Content\n")
        logging.info(f"Successfully wrote project structure to {output_file}")

    except IOError as e:
        logging.error(f"Error writing to output file {output_file}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


# --- Example Usage ---
if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = "/Users/ted/Documents/Studying/LLM/Projects/LangGraphLearning/rag_chatbot_project"
    # Assume the project root is one level up from the script's directory
    # ADJUST THIS PATH if your project structure is different
    project_root_dir = "/Users/ted/Documents/Studying/LLM/Projects/LangGraphLearning/rag_chatbot_project"

    # Or specify an absolute path directly:
    # project_root_dir = "/path/to/your/project"

    output_filename = "project_snapshot.txt"
    output_filepath = os.path.join(
        script_dir, output_filename
    )  # Save in the same dir as the script

    print(f"Project Directory: {project_root_dir}")
    print(f"Output File:       {output_filepath}")

    # Define specific patterns to exclude (add more as needed)
    custom_exclude_patterns = [
        ".git*",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.log",
        ".env",
        "venv",
        ".venv",
        "node_modules",
        "dist",
        "build",
        "*.egg-info",
        ".DS_Store",
        ".idea",
        ".vscode",
        "app.log",
        "setup.sh",
        "requirements.txt",
        output_filename,  # Exclude the output file itself
    ]

    # Define specific patterns to include (optional)
    # If you only want Python and text files:
    # custom_include_patterns = ["*.py", "*.txt", "*.md", "*.toml"]
    custom_include_patterns = None  # Include everything not excluded

    # Set a max file size in KB (e.g., 100KB) or None to include all sizes
    max_size_kb = 100

    if os.path.exists(project_root_dir):
        write_project_to_txt(
            project_dir=project_root_dir,
            output_file=output_filepath,
            exclude_patterns=custom_exclude_patterns,
            include_patterns=custom_include_patterns,
            max_file_size_kb=max_size_kb,
        )
        print("Done.")
    else:
        print(f"Error: Project directory '{project_root_dir}' does not exist.")
