import streamlit as st
import os
import hashlib
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# =========================================================
# SETUP
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="Nexa AI",
    page_icon="✦",
    layout="wide"
)

# =========================================================
# CHROMADB
# =========================================================

db = chromadb.PersistentClient(path="./chroma_db")

brain = db.get_or_create_collection("Nexa")
memory = db.get_or_create_collection("Nexa_chat")


# =========================================================
# DOCUMENT READING
# =========================================================

def read_file(file):

    if file.name.lower().endswith(".pdf"):

        reader = PdfReader(file)

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    elif file.name.lower().endswith(".txt"):

        return file.read().decode(
            "utf-8",
            errors="ignore"
        )

    return ""


# =========================================================
# SPLIT DOCUMENT INTO CHUNKS
# =========================================================

def chunk_text(text, chunk_size=1500):

    text = text.replace("\x00", " ")
    text = " ".join(text.split())

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            i:i + chunk_size
        ].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


# =========================================================
# STORE DOCUMENT
# =========================================================

def store_document(text, filename):

    if not text.strip():
        return 0

    chunks = chunk_text(text)

    if not chunks:
        return 0

    document_hash = hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):

        ids.append(
            f"{document_hash}_{i}"
        )

        documents.append(chunk)

        metadatas.append({
            "source": filename,
            "chunk": i
        })

    # Avoid adding the same document twice
    existing = brain.get(ids=ids)

    existing_ids = set(
        existing.get("ids", [])
    )

    new_ids = []
    new_documents = []
    new_metadatas = []

    for i in range(len(ids)):

        if ids[i] not in existing_ids:

            new_ids.append(ids[i])
            new_documents.append(
                documents[i]
            )
            new_metadatas.append(
                metadatas[i]
            )

    if new_ids:

        brain.add(
            ids=new_ids,
            documents=new_documents,
            metadatas=new_metadatas
        )

    return len(chunks)


# =========================================================
# SEARCH DOCUMENTS
# =========================================================

def search_documents(
    question,
    number=3
):

    if brain.count() == 0:
        return []

    try:

        results = brain.query(
            query_texts=[question],
            n_results=min(
                number,
                brain.count()
            )
        )

        documents = results.get(
            "documents"
        )

        if not documents:
            return []

        return documents[0]

    except Exception:
        return []


# =========================================================
# MEMORY
# =========================================================

def remember_exchange(
    question,
    answer
):

    exchange_id = hashlib.md5(
        f"{question}{answer}".encode(
            "utf-8"
        )
    ).hexdigest()

    memory.add(
        ids=[exchange_id],
        documents=[
            f"User: {question}\nNexa: {answer}"
        ],
        metadatas=[
            {
                "type": "conversation"
            }
        ]
    )


def recall_memory(
    question,
    number=2
):

    if memory.count() == 0:
        return []

    try:

        results = memory.query(
            query_texts=[question],
            n_results=min(
                number,
                memory.count()
            )
        )

        documents = results.get(
            "documents"
        )

        if not documents:
            return []

        return documents[0]

    except Exception:
        return []


# =========================================================
# API
# =========================================================

API_KEY = os.getenv(
    "AI_TOKEN"
)

if not API_KEY:

    st.error(
        "AI_TOKEN not found in .env"
    )

    st.stop()


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY
)


# =========================================================
# NEXA SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are Nexa AI, a helpful, intelligent and friendly
general-purpose AI assistant.

You help users with:

- General questions
- Studying
- Writing
- Coding
- Brainstorming
- Explaining concepts
- Document analysis
- Summarization

DOCUMENT RULES:

When relevant information is provided from the
Nexa document database, use it as your main source.

Do not invent information that is not supported
by the provided documents.

If the user asks a question specifically about an
uploaded document and the answer cannot be found
in the document database, say:

"I don't have enough information in my document
database to answer that."

You can summarize, explain, compare and analyze
information from uploaded documents.

Give clear and useful answers.

Adapt your explanation to the user's level.

Use examples when helpful.

Use emojis naturally when appropriate,
but do not overuse them.
"""


# =========================================================
# UI
# =========================================================

st.title("✦ Nexa AI")
st.subheader("Your personal AI assistant")


# =========================================================
# SIDEBAR
# =========================================================
st.markdown("""
<style>

.nexa-header {
    padding: 25px 30px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #0E171F 0%,
        #0A1118 50%,
        #07110D 100%
    );
    border: 1px solid #1E2B35;
    box-shadow: 0 10px 40px rgba(0, 230, 118, 0.08);
    margin-bottom: 25px;
}

.nexa-title {
    font-size: 42px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

.nexa-title span {
    color: #00E676;
}

.nexa-subtitle {
    color: #9AA7B5;
    font-size: 16px;
    margin-top: 5px;
}

</style>

<div class="nexa-header">

<div class="nexa-title">
✦ <span>Nexa</span> AI
</div>

<div class="nexa-subtitle">
Your intelligent personal AI assistant
</div>

</div>
""", unsafe_allow_html=True)


with st.sidebar:

    st.header("⚙️ AI Settings")

    name = st.text_input(
        "👤 Your name",
        placeholder="Enter your name"
    )

    mode = st.selectbox(
        "🤖 AI mode",
        [
            "General Assistant",
            "Study Tutor",
            "Coding Assistant",
            "Writing Assistant",
            "Creative Assistant"
        ]
    )

    response_style = st.selectbox(
        "✍️ Response style",
        [
            "Balanced",
            "Concise",
            "Detailed",
            "Step-by-step"
        ]
    )

    creativity = st.slider(
        "🎨 Creativity",
        0.0,
        1.0,
        0.7,
        0.1
    )

    remember_documents = st.slider(
        "📚 Documents to remember",
        0,
        5,
        3
    )

    recall = st.slider(
        "🧠 Memories to recall",
        0,
        5,
        2
    )

    web_search = st.toggle(
        "🌐 Allow web search",
        value=True
    )

    show_sources = st.toggle(
        "🔗 Show sources",
        value=True
    )

    st.divider()

    st.subheader("🛠️ Chat")

    if st.button(
        "🗑️ Clear conversation"
    ):

        st.session_state.messages = []

        st.rerun()

    if st.button(
        "💭 Forget long-term memory"
    ):

        try:

            db.delete_collection(
                "Nexa_chat"
            )

            memory = (
                db.get_or_create_collection(
                    "Nexa_chat"
                )
            )

        except Exception:
            pass

        st.rerun()


# =========================================================
# DOCUMENT UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload a document",
    type=["pdf", "txt"]
)


if uploaded_file:

    try:

        text = read_file(
            uploaded_file
        )

        if text.strip():

            chunks = store_document(
                text,
                uploaded_file.name
            )

            st.success(
                f"📚 {uploaded_file.name} "
                f"loaded successfully!"
            )

            st.caption(
                f"{chunks} document chunks "
                f"stored in ChromaDB."
            )

        else:

            st.error(
                "I couldn't extract text "
                "from this document."
            )

    except Exception as e:

        st.error(
            f"Error reading document: {e}"
        )


# =========================================================
# CHAT MEMORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

prompt = st.chat_input(
    "Message Nexa..."
)


if prompt:

    # -----------------------------------------------------
    # SHOW USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)


    # -----------------------------------------------------
    # SEARCH DOCUMENTS
    # -----------------------------------------------------

    document_results = search_documents(
        prompt,
        remember_documents
    )


    # -----------------------------------------------------
    # SEARCH MEMORY
    # -----------------------------------------------------

    memory_results = recall_memory(
        prompt,
        recall
    )


    # -----------------------------------------------------
    # BUILD DOCUMENT CONTEXT
    # -----------------------------------------------------

    if document_results:

        document_context = "\n\n".join(
            document_results
        )

    else:

        document_context = (
            "No relevant document "
            "information was found."
        )


    # -----------------------------------------------------
    # BUILD MEMORY CONTEXT
    # -----------------------------------------------------

    if memory_results:

        memory_context = "\n\n".join(
            memory_results
        )

    else:

        memory_context = (
            "No relevant previous "
            "memories were found."
        )


    # -----------------------------------------------------
    # PROTECT AGAINST HUGE REQUESTS
    # -----------------------------------------------------

    document_context = (
        document_context[:12000]
    )

    memory_context = (
        memory_context[:5000]
    )


    # -----------------------------------------------------
    # CONTEXT
    # -----------------------------------------------------

    context_prompt = f"""
User name:
{name if name else "Not provided"}

AI mode:
{mode}

Response style:
{response_style}

RELEVANT DOCUMENT INFORMATION:

{document_context}

RELEVANT MEMORY:

{memory_context}
"""


    # -----------------------------------------------------
    # KEEP ONLY RECENT CHAT
    # -----------------------------------------------------

    recent_messages = (
        st.session_state.messages[-10:]
    )


    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "system",
            "content": context_prompt
        }

    ]

    messages.extend(
        recent_messages
    )


    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Nexa is thinking... ✦"
        ):

            try:

                response = (
                    client.chat.completions.create(
                        model="groq/compound",
                        messages=messages,
                        temperature=creativity
                    )
                )

                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )

            except Exception as e:

                answer = (
                    f"❌ Error: {e}"
                )

        st.markdown(answer)


    # -----------------------------------------------------
    # SAVE MEMORY
    # -----------------------------------------------------

    if not answer.startswith("❌"):

        try:

            remember_exchange(
                prompt,
                answer
            )

        except Exception:
            pass


    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


