import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(
    page_title="Nexa AI",
    page_icon="✦",
    layout="wide"
)

st.title("Nexa AI")
st.subheader("Your personal AI assistant")

with st.sidebar:
    st.header("⚙️ AI Settings")
    name = st.text_input("👤 Your name", placeholder="Enter your name")
    mode = st.selectbox("🤖 AI mode",["General Assistant", "Study Tutor", "Coding Assistant", "Writing Assistant", "Creative Assistant"])
    response_style = st.selectbox("✍️ Response style",["Balanced", "Concise", "Detailed", "Step-by-step"])
    creativity = st.slider("🎨 Creativity",0.0,1.0,0.7,0.1)
    remember = st.slider("🧠 Recent messages",0,20,6)
    recall = st.slider("💾 Long-term memory",0,10,3)
    web_search = st.toggle("🌐 Allow web search",value=True)
    show_sources = st.toggle("🔗 Show sources",value=True)

    st.divider()

    st.subheader("🛠️ Chat")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.button("💭 Forget long-term memory"):
        # Put your memory deletion code here
        st.rerun()


API_KEY = os.getenv("AI_TOKEN")

if not API_KEY:
    st.error("AI_TOKEN not found in .env")
    st.stop()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY
)

SYSTEM_PROMPT = """
You are Nexa AI, a helpful, intelligent and friendly general-purpose AI assistant.

Help the user with questions, studying, writing, coding, brainstorming,
explaining concepts, and everyday tasks.

Give clear, useful and accurate answers.
Adapt your explanations to the user's level.
Use emojis naturally when appropriate, but don't overuse them.
"""

st.title("✦ Nexa AI")
st.caption("Your personal AI assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Message Nexa...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(st.session_state.messages)

    with st.chat_message("assistant"):

        with st.spinner("Nexa is thinking... ✦"):

            response = client.chat.completions.create(
                model="groq/compound",
                messages=messages,
                temperature=0.7
            )

            answer = response.choices[0].message.content

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
