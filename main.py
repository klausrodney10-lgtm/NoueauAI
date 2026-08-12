import streamlit as st
import os
import hashlib
import chromadb
from tavily import TavilyClient
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
st.set_page_config(page_title="Nexa AI", page_icon="✦", layout="wide")

# CHROMADB
db = chromadb.PersistentClient(path="./chroma_db")
brain = db.get_or_create_collection("Nexa")
memory = db.get_or_create_collection("Nexa_chat")

# DOCUMENT READING
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
        return file.read().decode("utf-8", errors="ignore")
    return ""

# SPLIT DOCUMENT INTO CHUNKS
def chunk_text(text, chunk_size=1000):
    text = text.replace("\x00", " ")
    text = " ".join(text.split())
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# STORE DOCUMENT
def store_document(text, filename):
    if not text.strip():
        return 0
    chunks = chunk_text(text)
    if not chunks:
        return 0
    document_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"{document_hash}_{i}")
        documents.append(chunk)
        metadatas.append({"source": filename, "chunk": i})
    existing = brain.get(ids=ids)
    existing_ids = set(existing.get("ids", []))
    new_ids, new_documents, new_metadatas = [], [], []
    for i in range(len(ids)):
        if ids[i] not in existing_ids:
            new_ids.append(ids[i])
            new_documents.append(documents[i])
            new_metadatas.append(metadatas[i])
    if new_ids:
        brain.add(ids=new_ids, documents=new_documents, metadatas=new_metadatas)
    return len(chunks)

# SEARCH DOCUMENTS
def search_documents(question, number=3):
    if brain.count() == 0:
        return []
    try:
        results = brain.query(
            query_texts=[question],
            n_results=min(number, brain.count())
        )
        documents = results.get("documents")
        if not documents:
            return []
        return documents[0]
    except Exception:
        return []

# MEMORY
def remember_exchange(question, answer):
    exchange_id = hashlib.md5(
        f"{question}{answer}".encode("utf-8")
    ).hexdigest()
    memory.add(
        ids=[exchange_id],
        documents=[f"User: {question}\nNexa: {answer}"],
        metadatas=[{"type": "conversation"}]
    )

def recall_memory(question, number=2):
    if memory.count() == 0:
        return []
    try:
        results = memory.query(
            query_texts=[question],
            n_results=min(number, memory.count())
        )
        documents = results.get("documents")
        if not documents:
            return []
        return documents[0]
    except Exception:
        return []

# API
API_KEY = os.getenv("AI_TOKEN")
if not API_KEY:
    st.error("AI_TOKEN not found in .env")
    st.stop()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY
)

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_KEY) if TAVILY_KEY else None

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

# NEXA SYSTEM PROMPT
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
The web search results are the PRIMARY source for
CURRENT and RECENT information.

When the user asks about:
- today's information
- this year
- a recent match
- current football results
- current standings
- current competitions
- recent news
- current prices
- current events

you MUST use the provided CURRENT WEB INFORMATION.

Do NOT use old knowledge to fill missing information.
Do NOT mix information from different years or seasons.

If the user asks about 2026, do not answer with
results from 2021, 2022, 2023, 2024 or 2025 unless
the user specifically asks about those years.

If the web results do not contain enough information,
say:
"Je n'ai pas trouvé suffisamment d'informations
récentes dans les sources disponibles."

Do NOT invent or guess missing results.

Do NOT recommend that the user search the Internet
themselves if web results were already provided.

When possible, mention the source and date of the
information.

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

# UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0,230,118,.08), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(0,200,255,.05), transparent 25%),
        #070B0F;
    color: #F4F7FA;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1,h2,h3 {
    font-family: 'Orbitron', sans-serif !important;
}

p,div,span,label {
    font-family: 'Inter', sans-serif;
}

.nexa-header {
    padding: 30px 34px;
    border-radius: 24px;
    background:
        linear-gradient(135deg,#0E171F 0%,#091018 55%,#07110D 100%);
    border: 1px solid #1E2B35;
    box-shadow:
        0 20px 60px rgba(0,230,118,.07);
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
}

.nexa-header:after {
    content:"";
    position:absolute;
    width:260px;
    height:260px;
    right:-100px;
    top:-120px;
    background:rgba(0,230,118,.08);
    border-radius:50%;
    filter:blur(45px);
}

.nexa-title {
    font-family:'Orbitron',sans-serif;
    font-size:44px;
    font-weight:800;
    margin:0;
    letter-spacing:-1px;
}

.nexa-title span {
    color:#00E676;
    text-shadow:0 0 25px rgba(0,230,118,.25);
}

.nexa-subtitle {
    color:#8F9CAB;
    font-size:15px;
    margin-top:6px;
}

.status {
    display:flex;
    align-items:center;
    gap:8px;
    color:#7F8A96;
    font-size:11px;
    letter-spacing:1px;
    margin-top:15px;
}

.status-dot {
    width:8px;
    height:8px;
    border-radius:50%;
    background:#00E676;
    box-shadow:0 0 14px rgba(0,230,118,.9);
}

.card {
    background:linear-gradient(135deg,#0E171F,#091018);
    border:1px solid #1E2B35;
    border-radius:18px;
    padding:20px;
    box-shadow:0 10px 35px rgba(0,230,118,.05);
    transition:.2s;
}

.card:hover {
    border-color:#00E676;
    transform:translateY(-2px);
}

.card-title {
    color:#7F8C99;
    font-size:11px;
    letter-spacing:1.5px;
    font-weight:700;
}

.card-value {
    color:#F4F7FA;
    font-family:'Orbitron',sans-serif;
    font-size:24px;
    font-weight:800;
    margin-top:7px;
}

.section-title {
    font-family:'Orbitron',sans-serif;
    color:#E8EDF2;
    font-size:14px;
    font-weight:700;
    letter-spacing:1.5px;
    margin-top:30px;
    margin-bottom:14px;
}

.stTextInput input,
.stSelectbox div[data-baseweb="select"] {
    background:#0D1318 !important;
    border:1px solid #26343D !important;
    border-radius:12px !important;
    color:white !important;
}

.stTextInput input:focus {
    border-color:#00E676 !important;
    box-shadow:0 0 0 1px #00E676 !important;
}

.stButton button {
    width:100%;
    border-radius:12px;
    border:1px solid #26343D;
    background:#10171C;
    color:white;
    font-weight:700;
    transition:.2s;
}

.stButton button:hover {
    border-color:#00E676;
    transform:translateY(-2px);
}

[data-testid="stChatMessage"] {
    border-radius:18px;
    border:1px solid rgba(255,255,255,.06);
    background:rgba(13,19,24,.72);
    padding:8px;
    margin-bottom:12px;
}

[data-testid="stChatInput"] {
    border-radius:18px !important;
}

[data-testid="stFileUploader"] {
    background:rgba(13,19,24,.7);
    border-radius:18px;
    border:1px dashed #2A3942;
    padding:10px;
}

.footer {
    text-align:center;
    color:#4E5A64;
    font-size:11px;
    margin-top:45px;
    padding-top:20px;
    border-top:1px solid rgba(255,255,255,.05);
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="nexa-header">
    <div class="nexa-title">✦ <span>Nexa</span> AI</div>
    <div class="nexa-subtitle">Your intelligent personal AI assistant</div>
    <div class="status">
        <div class="status-dot"></div>
        SYSTEM ONLINE • MEMORY ACTIVE • WEB READY
    </div>
</div>
""", unsafe_allow_html=True)

# DASHBOARD
col1,col2,col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">🧠 MEMORY</div>
        <div class="card-value">ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">📚 DOCUMENTS</div>
        <div class="card-value">READY</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">🌐 WEB</div>
        <div class="card-value">ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

# SETTINGS
with st.sidebar:
    st.header("⚙️ AI Settings")
    name = st.text_input("👤 Your name", placeholder="Enter your name")
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
        ["Balanced","Concise","Detailed","Step-by-step"]
    )
    creativity = st.slider("🎨 Creativity",0.0,1.0,0.7,0.1)
    remember_documents = st.slider("📚 Documents to remember",0,2,1)
    recall = st.slider("🧠 Memories to recall",0,5,2)
    web_search = st.toggle("🌐 Allow web search",value=True)
    show_sources = st.toggle("🔗 Show sources",value=True)
    st.divider()
    st.subheader("🛠️ Chat")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.button("💭 Forget long-term memory"):
        try:
            db.delete_collection("Nexa_chat")
            memory = db.get_or_create_collection("Nexa_chat")
        except Exception:
            pass
        st.rerun()

# DOCUMENTS
st.markdown(
    '<div class="section-title">📚 KNOWLEDGE BASE</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "📄 Upload a document",
    type=["pdf","txt"]
)

if uploaded_file:
    try:
        text = read_file(uploaded_file)
        if text.strip():
            chunks = store_document(text,uploaded_file.name)
            st.success(f"📚 {uploaded_file.name} loaded successfully!")
            st.caption(f"{chunks} document chunks stored in ChromaDB.")
        else:
            st.error("I couldn't extract text from this document.")
    except Exception as e:
        st.error(f"Error reading document: {e}")

# CHAT MEMORY
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# CHAT INPUT
prompt = st.chat_input("Message Nexa... ✦")

if prompt:
    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    document_results = search_documents(
        prompt,
        remember_documents
    )

    memory_results = recall_memory(
        prompt,
        recall
    )

    web_context = ""

    if web_search:
        web_context = search_web(prompt)

    document_context = (
        "\n\n".join(document_results)
        if document_results
        else "No relevant document information was found."
    )

    memory_context = (
        "\n\n".join(memory_results)
        if memory_results
        else "No relevant previous memories were found."
    )

    document_context = document_context[:4000]
    memory_context = memory_context[:1000]

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
{web_context if web_context else "No web information available."}
"""

    recent_messages = st.session_state.messages[-4:]

    messages = [
        {
            "role":"system",
            "content":SYSTEM_PROMPT
        },
        {
            "role":"system",
            "content":context_prompt
        }
    ]

    messages.extend(recent_messages)

    with st.chat_message("assistant"):
        with st.spinner("Nexa is thinking... ✦"):
            try:
                print("Document characters:",len(document_context))
                print("Memory characters:",len(memory_context))
                print("Recent messages:",len(recent_messages))
                print(
                    "Total message characters:",
                    sum(len(str(m["content"])) for m in messages)
                )

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=creativity
                )

                answer = response.choices[0].message.content

            except Exception as e:
                answer = f"❌ Error: {e}"

        st.markdown(answer)

    if not answer.startswith("❌"):
        try:
            remember_exchange(prompt,answer)
        except Exception:
            pass

    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })

# FOOTER
st.markdown("""
<div class="footer">
    ✦ Nexa AI • Intelligent Personal Assistant • Built with AI
</div>
""", unsafe_allow_html=True)






