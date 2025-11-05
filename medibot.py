import os
# Streamlit UI
import streamlit as st
import time
# Load environment variables from .env (so GROQ_API_KEY in project root is available)
from dotenv import load_dotenv
from pathlib import Path
# ensure we load the .env next to this file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq


## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def build_chain_type_kwargs(chain_type: str, base_prompt: PromptTemplate):
    """Return appropriate chain_type_kwargs for LangChain's RetrievalQA.from_chain_type.

    Different chain types expect different kwarg names:
      - 'stuff' expects 'prompt'
      - 'map_reduce' expects 'map_prompt' and 'combine_prompt'
      - 'refine' expects 'question_prompt' and 'refine_prompt'

    We build sensible default prompts for map/reduce and refine so callers can
    switch chain types without causing Pydantic "extra inputs" errors.
    """
    if chain_type == "stuff":
        return {"prompt": base_prompt}

    if chain_type == "map_reduce":
        map_prompt = PromptTemplate(
            template=("Read the following passage and write a concise summary of the key facts.\n\n{text}\n"),
            input_variables=["text"],
        )
        combine_prompt = PromptTemplate(
            template=("Using the summaries below, answer the question.\n\nSummaries:\n{summaries}\n\nQuestion: {question}\nAnswer:"),
            input_variables=["summaries", "question"],
        )
        return {"map_prompt": map_prompt, "combine_prompt": combine_prompt}

    if chain_type == "refine":
        question_prompt = PromptTemplate(
            template=("Answer the question based on the context below. If the context is not sufficient, say you don't know.\n\nContext:\n{context_str}\n\nQuestion: {question}\nAnswer:"),
            input_variables=["context_str", "question"],
        )
        refine_prompt = PromptTemplate(
            template=("Given the existing answer and the new context, refine the answer to be more accurate.\n\nExisting answer:\n{existing_answer}\n\nNew context:\n{context_str}\n\nQuestion: {question}\nRefined answer:"),
            input_variables=["existing_answer", "context_str", "question"],
        )
        return {"question_prompt": question_prompt, "refine_prompt": refine_prompt}

    # Default fallback: treat as 'stuff'
    return {"prompt": base_prompt}


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        # concise synthesis prompt: request a single concise answer and one best source
        CUSTOM_PROMPT_TEMPLATE = (
            "Using only the provided CONTEXT, give a single concise factual answer to the QUESTION. "
            "If the context does not contain the answer, say you don't know. Do not hallucinate. "
            "Answer in one paragraph (max 60 words). Then list the single best source (filename + page) in one line.\n\n"
            "Context: {context}\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        
        #HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # PAID
        #HF_TOKEN=os.environ.get("HF_TOKEN")  

        #TODO: Create a Groq API key and add it to .env file
        
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            # Primary: Groq-hosted model
            groq_llm = ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.0,
                groq_api_key=os.environ.get("GROQ_API_KEY"),
            )

            # Use 'refine' to synthesize a single concise answer from multiple retrieved chunks
            chain_type_name = "refine"  # better synthesis for single exact answer
            base_prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
            chain_kwargs = build_chain_type_kwargs(chain_type_name, base_prompt)

            qa_chain = RetrievalQA.from_chain_type(
                llm=groq_llm,
                chain_type=chain_type_name,
                # retrieve a few (k) relevant chunks, refine will synthesize into one answer
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs=chain_kwargs,
            )

            # Try invoke with exponential backoff retries. If Groq is still unavailable
            # after retries (503 / over capacity), fall back to a HuggingFace endpoint LLM
            # (if configured). This prevents the UI from failing outright when Groq is
            # temporarily overloaded.
            max_retries = 4
            response = None
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = qa_chain.invoke({'query': prompt})
                    break
                except Exception as e:
                    last_exc = e
                    msg = str(e).lower()
                    # detect common Groq over-capacity message or 503
                    if 'over capacity' in msg or '503' in msg or 'service unavailable' in msg:
                        if attempt < max_retries:
                            wait = 2 ** (attempt - 1)
                            st.warning(f"Groq model over capacity — retrying in {wait}s (attempt {attempt}/{max_retries})...")
                            time.sleep(wait)
                            continue
                        # after retries, fallback
                        st.warning("Groq model still unavailable after retries — attempting fallback LLM.")
                        hf_token = os.environ.get("HF_TOKEN")
                        hf_repo = os.environ.get("HF_REPO_ID", None)
                        if hf_repo is None:
                            # If no HF repo configured, try a conservative default (will work if HF_TOKEN present)
                            hf_repo = "google/flan-t5-large"

                        try:
                            hf_llm = load_llm(hf_repo, hf_token)
                            # For the fallback LLM, reuse the same chain_type_name and prompt setup
                            hf_chain_kwargs = build_chain_type_kwargs("stuff", set_custom_prompt(CUSTOM_PROMPT_TEMPLATE))
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=hf_llm,
                                chain_type="stuff",
                                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                                return_source_documents=True,
                                chain_type_kwargs=hf_chain_kwargs,
                            )
                            response = qa_chain.invoke({'query': prompt})
                            break
                        except Exception as hf_e:
                            # If fallback also fails, show helpful message and break
                            st.error(f"Both Groq and fallback LLM failed: {hf_e}")
                            raise
                    else:
                        # non-capacity error — re-raise so outer except can show it
                        raise

            if response is None:
                # If we exit loop without a response, raise the last exception to be handled below
                raise last_exc if last_exc is not None else RuntimeError("No response from LLM")

            result = response["result"]
            source_documents = response.get("source_documents", [])

            # debug output removed — metadata will only be used to build formatted citations

            # Format source documents into a readable citation list instead of
            # dumping the raw Document objects. Each entry shows source, page
            # (if available) and a short snippet from the page_content.
            def format_source_docs(docs, max_snippet=300, top_n=1):
                from pathlib import Path
                lines = []

                # deduplicate by filename and keep top_n
                seen = set()
                filtered = []
                for doc in docs:
                    # normalize metadata safely
                    if hasattr(doc, "metadata"):
                        meta = doc.metadata or {}
                    elif isinstance(doc, dict):
                        meta = doc.get("metadata", {}) or {}
                    else:
                        meta = {}

                    # determine a filename-like source
                    source_val = (
                        meta.get("source")
                        or meta.get("file")
                        or meta.get("filename")
                        or meta.get("file_path")
                        or meta.get("source_id")
                        or meta.get("doc_id")
                        or meta.get("title")
                    )
                    if source_val:
                        try:
                            fname = Path(str(source_val)).name
                        except Exception:
                            fname = str(source_val)
                    else:
                        fname = getattr(doc, "source", None) or getattr(doc, "title", None) or ""
                    fname = str(fname) if fname else "unknown_source"

                    if fname in seen:
                        continue
                    seen.add(fname)
                    filtered.append((doc, meta, fname))
                    if len(filtered) >= top_n:
                        break

                for i, (doc, meta, fname) in enumerate(filtered, start=1):
                    page = meta.get("page") or meta.get("page_label") or meta.get("page_number")
                    # robustly get text
                    snippet = getattr(doc, "page_content", None) or getattr(doc, "content", None)
                    if snippet is None and isinstance(doc, dict):
                        snippet = doc.get("page_content") or doc.get("content") or ""
                    snippet = " ".join(str(snippet).split())[:max_snippet].strip()
                    page_info = f" (page {page})" if page else ""
                    lines.append(f"{i}. Source: {fname}{page_info}\n   Snippet: {snippet}")

                return "\n\n".join(lines) if lines else "No source documents."

            formatted_sources = format_source_docs(source_documents, top_n=1)
            result_to_show = result + "\n\nSource Docs:\n" + formatted_sources
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()