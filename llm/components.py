import logging
import os
import re
import tempfile
import json
import uuid
from typing import Any, Dict, List, Optional, Union
import requests

from chatbot.config import settings

from pinecone import Pinecone


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    ArxivLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
    WebBaseLoader,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

try:
    from pinecone.openapi_support.exceptions import PineconeApiException
except ImportError:
    PineconeApiException = Exception

from streamlit.runtime.uploaded_file_manager import UploadedFile


logger = logging.getLogger("RAG_Chatbot_App")


def _load_arxiv_source(paper_id: str) -> List[Document]:
    """Loads a document from arXiv using its ID."""
    logger.debug(f"Using ArxivLoader for ID: {paper_id}")
    try:
        loader = ArxivLoader(
            query=paper_id, load_max_docs=1, load_all_available_meta=True
        )
        docs = loader.load()
        logger.debug(
            f"Finished loading from arXiv: {paper_id}. Documents loaded: {len(docs)}"
        )
        return docs
    except Exception as e:
        logger.error(f"ArxivLoader failed for ID '{paper_id}': {e}", exc_info=True)
        return []


def _load_pdf_from_url(url: str, temp_files_list: List[str]) -> List[Document]:
    """Downloads a PDF from a URL and loads it using PyPDFLoader."""
    logger.debug(f"Downloading direct PDF link: {url}")
    file_path = None
    docs = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            file_path = tmp_file.name
        temp_files_list.append(file_path)
        logger.debug(f"Downloaded PDF to temporary file: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for doc in docs:
            if "source" not in doc.metadata:
                doc.metadata["source"] = url
        logger.debug(
            f"Finished loading from downloaded PDF: {url}. Pages loaded: {len(docs)}"
        )
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Failed to download PDF URL '{url}': {req_err}", exc_info=True)
    except Exception as pdf_err:
        logger.error(
            f"Failed to process downloaded PDF '{url}': {pdf_err}", exc_info=True
        )

    return docs


def _load_web_html(url: str) -> List[Document]:
    """Loads content from a general web URL, likely HTML."""
    logger.debug(f"Assuming HTML, using WebBaseLoader for URL: {url}")
    try:
        loader = WebBaseLoader([url])
        docs = loader.load()
        logger.debug(
            f"Finished loading from WebBaseLoader URL: {url}. Documents loaded: {len(docs)}"
        )
        return docs
    except Exception as web_err:
        logger.error(f"WebBaseLoader failed for URL '{url}': {web_err}", exc_info=True)
        return []


def _load_local_file(file_path: str) -> List[Document]:
    """Loads a document from a local file path (PDF or other)."""
    logger.debug(f"Loading from local file path: {file_path}")
    try:
        _, ext = os.path.splitext(file_path)
        if ext.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            logger.warning(f"Local file is not PDF, trying Unstructured: {file_path}")
            loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        logger.debug(
            f"Finished loading from local file path: {file_path}. Documents loaded: {len(docs)}"
        )
        return docs
    except Exception as e:
        logger.error(f"Failed loading local file '{file_path}': {e}", exc_info=True)
        return []


def _load_uploaded_file(
    uploaded_file: UploadedFile, temp_files_list: List[str]
) -> List[Document]:
    """Loads data from a Streamlit UploadedFile object."""
    file_name = uploaded_file.name
    logger.debug(f"Processing UploadedFile: {file_name}")
    file_path = None
    docs = []
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file_name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name
        temp_files_list.append(file_path)
        logger.debug(f"Saved UploadedFile to temp: {file_path}")
        _, ext = os.path.splitext(file_path)
        if ext.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            logger.warning(f"Uploaded file not PDF, trying Unstructured: {file_name}")
            loader = UnstructuredFileLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file_name
        logger.debug(
            f"Finished loading from UploadedFile temp: {file_name}. Documents loaded: {len(docs)}"
        )
    except Exception as e:
        logger.error(
            f"Failed processing UploadedFile '{file_name}': {e}", exc_info=True
        )

    return docs


def _clean_document_text(doc: Document) -> Document:
    """Normalizes whitespace in the document's page_content."""
    if isinstance(doc.page_content, str):
        text = doc.page_content

        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if cleaned_text != text:
            logger.debug(
                f"Cleaned whitespace in doc from source: {doc.metadata.get('source', 'N/A')[:50]}..."
            )
        return Document(page_content=cleaned_text, metadata=doc.metadata)
    else:
        logger.warning(
            f"Doc page_content is not a string, skipping cleaning. Type: {type(doc.page_content)}"
        )
        return doc


def _cleanup_temp_files(temp_files: List[str]):
    """Removes temporary files created during loading."""
    logger.debug(f"Cleaning up {len(temp_files)} temporary files...")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            logger.debug(f"Deleted temporary file: {temp_file}")
        except Exception as e:
            logger.warning(f"Could not delete temporary file {temp_file}: {e}")


def load_and_split(sources: List[Union[str, UploadedFile]]) -> List[Document]:
    """
    Loads, cleans, and splits documents from various sources.
    """
    logger.info(f"Starting load, clean, and split for {len(sources)} sources.")
    all_loaded_docs = []
    temp_files_to_clean = []
    url_pattern = re.compile(r"https?://\S+")
    arxiv_pattern = re.compile(
        r"https?://arxiv\.org/(?:abs|pdf)/(\d+\.\d+(?:v\d+)?)(?:\.pdf)?$"
    )

    for source_num, source in enumerate(sources):
        docs_from_source = []
        try:
            if isinstance(source, str) and url_pattern.match(source):
                arxiv_match = arxiv_pattern.match(source)
                if arxiv_match:
                    docs_from_source = _load_arxiv_source(arxiv_match.group(1))
                elif source.lower().endswith(".pdf"):
                    docs_from_source = _load_pdf_from_url(source, temp_files_to_clean)
                else:
                    docs_from_source = _load_web_html(source)
            elif isinstance(source, str):
                docs_from_source = _load_local_file(source)
            elif isinstance(source, UploadedFile):
                docs_from_source = _load_uploaded_file(source, temp_files_to_clean)
            else:
                logger.warning(
                    f"Source {source_num+1}: Unknown type '{type(source)}', skipping."
                )

            cleaned_docs = [_clean_document_text(doc) for doc in docs_from_source]
            all_loaded_docs.extend(cleaned_docs)

        except Exception as e:

            logger.error(
                f"Error processing source {source_num+1} ('{str(source)[:50]}...'): {e}",
                exc_info=True,
            )

    if not all_loaded_docs:
        logger.warning(
            "No documents were loaded successfully after processing all sources."
        )
        _cleanup_temp_files(temp_files_to_clean)
        return []

    logger.info(f"Total documents loaded and cleaned: {len(all_loaded_docs)}")

    logger.debug(
        f"Splitting {len(all_loaded_docs)} documents (chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})..."
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    try:
        split_docs = text_splitter.split_documents(all_loaded_docs)
        logger.info(f"Split into {len(split_docs)} chunks.")
    except Exception as split_err:
        logger.error(f"Error splitting documents: {split_err}", exc_info=True)
        _cleanup_temp_files(temp_files_to_clean)
        return []

    if not split_docs:
        logger.warning("Splitting resulted in zero documents.")
        _cleanup_temp_files(temp_files_to_clean)
        return []

    if split_docs:
        logger.debug(
            f"First chunk content preview (cleaned): {split_docs[0].page_content[:300]}..."
        )
        logger.debug(f"Metadata of first chunk: {split_docs[0].metadata}")

    _cleanup_temp_files(temp_files_to_clean)
    return split_docs


def clean_metadata_for_pinecone(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cleans a metadata dictionary to be compatible with Pinecone.
    Removes Nones, truncates long strings, attempts conversion, checks total size.
    """
    if not metadata:
        return {}

    if "text" in metadata:
        logger.warning(
            "Removing unexpected 'text' key found in metadata during cleaning."
        )
        metadata = metadata.copy()
        del metadata["text"]

    cleaned_metadata = {}
    estimated_size = 0
    allowed_types = (str, bool, int, float)
    keys_to_prune_on_size_limit = ["summary", "authors"]

    keys_to_process = list(metadata.keys())

    for key in keys_to_process:
        value = metadata.get(key)
        if value is None:
            continue

        cleaned_value: Any = None

        if isinstance(value, allowed_types):
            if isinstance(value, str):
                cleaned_value = value[: settings.MAX_METADATA_FIELD_LENGTH]
                if len(value) > settings.MAX_METADATA_FIELD_LENGTH:
                    logger.warning(f"Truncated metadata string field '{key}'.")
            else:
                cleaned_value = value

        elif isinstance(value, list):
            str_list = []
            valid_list = True
            for item in value:
                if isinstance(item, str):
                    str_list.append(item[: settings.MAX_METADATA_FIELD_LENGTH])
                else:
                    valid_list = False
                    break
            if valid_list and str_list:
                cleaned_value = str_list
            else:
                logger.warning(
                    f"Skipping metadata list '{key}' (non-string item or empty)."
                )
                continue

        else:
            logger.warning(
                f"Metadata field '{key}' type {type(value)}. Converting to string."
            )
            try:
                cleaned_value = str(value)[: settings.MAX_METADATA_FIELD_LENGTH]
            except Exception:
                logger.warning(
                    f"Could not convert metadata '{key}' to string. Skipping."
                )
                continue

        if cleaned_value is not None:
            cleaned_metadata[key] = cleaned_value

    try:
        metadata_json_string = json.dumps(cleaned_metadata)
        estimated_size = len(metadata_json_string.encode("utf-8"))
    except Exception as json_err:
        logger.warning(
            f"Could not estimate metadata size using JSON for pruning: {json_err}"
        )
        estimated_size = settings.MAX_TOTAL_METADATA_BYTES + 1

    if estimated_size > settings.MAX_TOTAL_METADATA_BYTES:
        logger.warning(
            f"Estimated metadata size ({estimated_size} bytes) exceeds limit ({settings.MAX_TOTAL_METADATA_BYTES} bytes). Pruning..."
        )

        for prune_key in keys_to_prune_on_size_limit:
            if prune_key in cleaned_metadata:
                del cleaned_metadata[prune_key]
                logger.info(f"Pruned metadata field '{prune_key}'.")
                try:
                    estimated_size = len(json.dumps(cleaned_metadata).encode("utf-8"))
                except Exception:
                    pass
                if estimated_size <= settings.MAX_TOTAL_METADATA_BYTES:
                    break
        if estimated_size > settings.MAX_TOTAL_METADATA_BYTES:
            logger.error(
                f"Metadata size ({estimated_size}) STILL exceeds limit after pruning."
            )

    return cleaned_metadata


def get_pinecone_vectorstore(index_name: str) -> Optional[PineconeVectorStore]:
    """Initializes connection to an existing Pinecone index."""
    logger.info(f"Initializing connection to Pinecone index: {index_name}")
    if not index_name:
        logger.error("Pinecone index name not provided.")
        return None
    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)

        if not hasattr(vectorstore, "index_name"):
            vectorstore.index_name = index_name
        logger.info(f"Connected to Pinecone index '{index_name}' successfully.")
        return vectorstore
    except Exception as e:
        logger.error(
            f"Error initializing Pinecone VectorStore for index '{index_name}': {e}",
            exc_info=True,
        )
        return None


def embed_and_upsert(docs: List[Document], vectorstore: PineconeVectorStore) -> bool:
    batch = []
    total_size = 0

    def flush_batch():
        if batch:
            vectorstore.add_documents(batch)
            batch.clear()

    index_name = getattr(vectorstore, "index_name", "[Unknown Index]")
    print(f"index name: {index_name}")
    docs_to_upsert = []

    for doc in docs:
        cleaned_meta = clean_metadata_for_pinecone(doc.metadata or {})
        batch.append(Document(page_content=doc.page_content, metadata=cleaned_meta))
        # print(doc)
        total_size += len(doc.page_content.encode("utf-8"))

        if total_size >= 3 * 1024 * 1024 or len(batch) >= settings.PINECONE_BATCH_SIZE:
            flush_batch()
            total_size = 0

    flush_batch()
    return True


# def embed_and_upsert(docs: List[Document], vectorstore: PineconeVectorStore) -> bool:
#     """
#     Embeds documents and upserts them ONE BY ONE to Pinecone after cleaning metadata.
#     Uses direct index.upsert for debugging. Catches correct Pinecone exception.
#     """
#     if not docs: logger.warning("No documents to embed and upsert."); return False
#     if not vectorstore: logger.error("Invalid vectorstore provided."); return False
#     # Safely access the index object
#     try:
#         index = vectorstore.index
#         if index is None: raise AttributeError("vectorstore.index is None")
#     except AttributeError:
#          logger.error("Cannot access the raw Pinecone index object via vectorstore.index.")
#          return False
#     # Safely access the embedding function
#     try:
#         # Correct attribute is likely embeddings, not embedding
#         embedder = vectorstore.embeddings
#         if embedder is None: raise AttributeError("vectorstore.embeddings is None")
#     except AttributeError:
#         logger.error("Cannot access the embedding function via vectorstore.embeddings. Initializing fallback.")
#         try:
#             # Ensure API key is available for fallback
#             if not settings.OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY needed for fallback embedder")
#             embedder = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY)
#         except Exception as init_err:
#              logger.error(f"Failed to initialize fallback OpenAIEmbeddings: {init_err}", exc_info=True)
#              return False


#     index_name = getattr(vectorstore, 'index_name', '[Unknown Index]')
#     logger.info(f"Starting MANUAL embed/upsert of {len(docs)} chunks one-by-one to index '{index_name}'...")

#     success_count = 0
#     fail_count = 0
#     # Use the namespace associated with the vectorstore object if available
#     # Note: _namespace might be internal, check Langchain docs for public access method if needed
#     namespace = getattr(vectorstore, '_namespace', None)
#     logger.info(f"Using Pinecone namespace: '{namespace}'")

#     for i, doc in enumerate(docs):
#         logger.debug(f"Processing document {i+1}/{len(docs)}...")

#         # 1. Clean Metadata
#         cleaned_metadata = clean_metadata_for_pinecone(doc.metadata or {})
#         logger.debug(f"Cleaned metadata for doc {i}: {cleaned_metadata}")

#         # 2. Generate Embedding
#         try:
#             # Use embed_query for single text embedding
#             vector = embedder.embed_query(doc.page_content)
#             logger.debug(f"Generated embedding for doc {i} (size: {len(vector)})")
#         except Exception as embed_err:
#             logger.error(f"Failed to generate embedding for doc {i}: {embed_err}", exc_info=True)
#             fail_count += 1
#             continue # Skip to the next document

#         # 3. Generate ID
#         doc_id = uuid.uuid4().hex
#         logger.debug(f"Generated ID for doc {i}: {doc_id}")

#         # 4. Prepare vector tuple for Pinecone upsert
#         vector_tuple = (doc_id, vector, cleaned_metadata)

#         # 5. Upsert SINGLE vector
#         try:
#             logger.debug(f"Attempting to upsert vector for doc {i} (ID: {doc_id}) into namespace '{namespace}'...")
#             # Use the raw index object's upsert method
#             upsert_response = index.upsert(vectors=[vector_tuple], namespace=namespace)
#             logger.debug(f"Upsert response for doc {i}: {upsert_response}")
#             success_count += 1
#         except PineconeApiException as pinecone_err:
#             # Catch the CORRECTED Pinecone specific API errors
#             logger.error(f"Pinecone API Error upserting doc {i} (ID: {doc_id}): {pinecone_err}", exc_info=True)
#             logger.error(f"--- Failing vector metadata (doc {i}): {cleaned_metadata}")
#             logger.error(f"--- Failing vector content preview (doc {i}): {doc.page_content[:300]}...")
#             fail_count += 1
#             # break # Optional: Stop on first error
#         except Exception as other_err:
#             # Catch other potential errors during upsert
#             logger.error(f"Unexpected Error upserting doc {i} (ID: {doc_id}): {other_err}", exc_info=True)
#             fail_count += 1
#             # break # Optional: Stop on first error

#     logger.info(f"Manual upsert process finished. Successful: {success_count}, Failed: {fail_count}")
#     # Return True only if all documents were successfully upserted
#     return fail_count == 0


def similarity_search(
    query: str, vectorstore: PineconeVectorStore, k: int = settings.TOP_K
) -> str:
    """Performs similarity search and returns formatted context string."""
    if not vectorstore:
        logger.error("Invalid vectorstore for similarity_search.")
        return ""
    index_name = getattr(vectorstore, "index_name", "[Unknown Index]")
    logger.info(
        f"Performing similarity search (k={k}) on index '{index_name}' for query: '{query[:50]}...'"
    )
    try:
        retrieved_docs = vectorstore.similarity_search(query, k=k)
        if retrieved_docs:
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            logger.info(
                f"Similarity search found {len(retrieved_docs)} relevant chunks."
            )
            return context
        else:
            logger.info("Similarity search found no matching chunks.")
            return ""
    except Exception as e:
        logger.error(f"Error during similarity search: {e}", exc_info=True)
        return ""


def web_search(query: str, max_results: int = settings.TAVILY_MAX_RESULTS) -> str:
    """Performs web search using Tavily and returns formatted results string."""
    logger.info(
        f"Performing web search via Tavily (max_results={max_results}) for query: '{query[:50]}...'"
    )
    try:
        tavily_tool = TavilySearchResults(max_results=max_results)
        results = tavily_tool.invoke(query)
        if results and isinstance(results, list):
            formatted_results = "\n\n---\n\n".join(
                [
                    f"Source: {res.get('url', 'N/A')}\nContent: {res.get('content', '')}"
                    for res in results
                    if isinstance(res, dict)
                ]
            )
            logger.info(f"Web search found {len(results)} results.")
            return formatted_results
        else:
            logger.info("Web search found no results or invalid format.")
            return ""
    except Exception as e:
        logger.error(f"Error during Tavily search: {e}", exc_info=True)
        return ""


def get_llm() -> Optional[ChatOpenAI]:
    """Initializes and returns the ChatOpenAI LLM instance."""
    logger.debug("Initializing ChatOpenAI LLM...")
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, streaming=True)
        logger.info("ChatOpenAI LLM initialized.")
        return llm
    except Exception as e:
        logger.error(f"Error initializing ChatOpenAI: {e}", exc_info=True)
        return None


# If the information is not available in the context, state that you could not find the information in the provided documents.
# Answer politely and in detail. DO NOT make up information. Always answer in English.
def create_rag_prompt() -> ChatPromptTemplate:
    """Creates the prompt template for RAG (English)."""
    logger.debug("Creating prompt template for RAG.")
    template = """You are a helpful AI assistant. Your task is to answer the user's question based PRIMARILY on the provided context below.

Context:
{context}

Chat History:
{chat_history}

User Question: {query}

Your Answer (English):"""
    return ChatPromptTemplate.from_template(template)


def create_web_search_prompt() -> ChatPromptTemplate:
    """Creates the prompt template for web search results (English)."""
    logger.debug("Creating prompt template for Web Search.")
    template = """You are a helpful AI assistant. Your task is to answer the user's question based PRIMARILY on the web search results provided below.
Synthesize the information from the search results to provide a comprehensive answer.
If the information is not available in the search results, state that you could not find relevant information on the web.
Answer politely and in detail. DO NOT make up information. Always answer in English.

Web Search Results:
{context}

Chat History:
{chat_history}

User Question: {query}

Your Answer (English):"""
    return ChatPromptTemplate.from_template(template)


def create_generation_chain(llm: ChatOpenAI, prompt: ChatPromptTemplate) -> Any:
    """Creates a LangChain Runnable (LCEL chain) for generating responses."""
    logger.debug("Creating generation chain (prompt | llm | StrOutputParser).")
    if not llm or not prompt:
        logger.error("Invalid LLM or Prompt provided...")
        raise ValueError("LLM and Prompt are required...")
    return prompt | llm | StrOutputParser()


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Formats chat history into a readable string for the prompt (English)."""
    if not chat_history:
        return "No chat history."
    formatted = []
    for msg in chat_history:
        role = "User" if msg.get("role") == "user" else "Assistant AI"
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)
