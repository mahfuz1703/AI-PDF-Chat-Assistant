import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from chat_ui import CHAT_CSS, StreamHandler, build_messages_html


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vector_store, stream_handler):
    # Streaming LLM for the final answer
    answer_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        streaming=True,
        callbacks=[stream_handler],
    )
    # Non-streaming LLM for condensing follow-up questions
    condense_llm = ChatGroq(model="openai/gpt-oss-120b")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=answer_llm,
        condense_question_llm=condense_llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )


def main():
    load_dotenv()
    st.set_page_config(page_title="AI PDF Chat Assistant", page_icon="🤖", layout="wide")
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stream_handler" not in st.session_state:
        st.session_state.stream_handler = StreamHandler()

    # ── Sidebar ──────────────
    with st.sidebar:
        st.title("📄 PDF Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files and click **Process**",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("⚙️ Process", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDFs…"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversation_chain(
                        vector_store, st.session_state.stream_handler
                    )
                    st.session_state.messages = []
                st.success(f"✅ {len(pdf_docs)} PDF(s) processed!")

        if st.session_state.conversation:
            st.markdown("---")
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.experimental_rerun()

    # ── Main area ─────────────────
    st.title("🤖 AI PDF Chat Assistant")

    if not st.session_state.conversation:
        st.info("👈 Upload your PDF documents in the sidebar to get started.")
        return

    # Render order: messages → streaming slot → input form
    msg_slot = st.empty()
    stream_slot = st.empty()

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([9, 1])
        with col1:
            user_input = st.text_input(
                "message",
                placeholder="Ask anything about your PDFs…",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        prompt = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show updated history (with user message) immediately
        msg_slot.markdown(build_messages_html(st.session_state.messages), unsafe_allow_html=True)

        # Stream the response into stream_slot token by token
        st.session_state.stream_handler.reset(stream_slot)
        response = st.session_state.conversation({"question": prompt})
        answer = response["answer"]

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.experimental_rerun()
    else:
        msg_slot.markdown(build_messages_html(st.session_state.messages), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
