import streamlit as st
import os
import hashlib
import chromadb
from tavily import TavilyClient
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

def chunk_text(text, chunk_size=1000):

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

TAVILY_KEY = os.getenv("TAVILY_API_KEY")

if TAVILY_KEY:
    tavily = TavilyClient(api_key=TAVILY_KEY)
else:
    tavily = None

def search_web(question):

    if tavily is None:
        return ""

    try:
        results = tavily.search(
            query=question,
            max_results=3
        )

        web_context = ""

        for result in results.get("results", []):
            web_context += f"""
Title: {result.get("title", "")}
URL: {result.get("url", "")}
Information: {result.get("content", "")}

"""

        return web_context[:4000]

    except Exception as e:
        return f"Web search error: {e}"

# =========================================================
# NEXA SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are Nexa AI, a helpful, intelligent and friendly
general-purpose AI assistant.

LANGUAGE:
Always answer in the same language as the user.
If the user writes in French, answer in French.
If the user writes in English, answer in English.
If the user writes in Spanish, answer in Spanish.

You can help with:
- General questions
- Studying
- Writing
- Coding
- Brainstorming
- Explaining concepts
- Document analysis
- Current information
- Sports and football

WEB SEARCH RULES:

When CURRENT or RECENT information is needed,
use the CURRENT WEB INFORMATION provided to you.

This includes:
- Current football results
- Recent football matches
- Champions League results
- World Cup information
- League standings
- Recent news
- Current prices
- Recent events

IMPORTANT:

If CURRENT WEB INFORMATION is provided, use it
to answer the user's question.

Do NOT say:
"I don't have access to current information."

Do NOT tell the user to search the Internet themselves
when useful web information has already been provided.

Use the web information to create a direct answer.

If the web information does not contain the answer,
clearly say that the available web sources did not
provide enough information.

DOCUMENT RULES:

When relevant document information is provided,
use it to answer questions about uploaded documents.

Do not invent information from documents.

If the user asks about an uploaded document and the
answer cannot be found in the provided document
information, say:

"I don't have enough information in my document
database to answer that."

MEMORY:

Use relevant information from the provided memory
when it helps answer the user's question.

Do not reveal private memory unless it is relevant
to the user's request.

GENERAL BEHAVIOR:

Give direct, clear and useful answers.

For factual questions, prioritize reliable and recent
information.

When using web information, mention the source when
appropriate.

Use examples when helpful.

Use emojis naturally, but don't overuse them.
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
        2,
        1
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

    web_context = ""

    if web_search:
        web_context = search_web(prompt)

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
        document_context[:4000]
    )

    memory_context = (
        memory_context[:1000]
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

CURRENT WEB INFORMATION:

{web_context if web_context else "No web search was performed."}
"""


    # -----------------------------------------------------
    # KEEP ONLY RECENT CHAT
    # -----------------------------------------------------

    recent_messages = (
        st.session_state.messages[-4:]
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

                print("Document characters:", len(document_context))
                print("Memory characters:", len(memory_context))
                print("Recent messages:", len(recent_messages))
                print("Total message characters:",
                      sum(len(str(m["content"])) for m in messages))

                response = (
                    client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
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


