from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()
import streamlit as st

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# ── Domain definitions ────────────────────────────────────────────────────────
PERSONAS = {
    "🔬 Data Science Expert": {
        "system": (
            "You are an expert Data Scientist with deep knowledge of statistics, "
            "machine learning, deep learning, data wrangling, and tools like Python, "
            "pandas, scikit-learn, and TensorFlow. Explain concepts clearly with examples "
            "and code snippets when helpful. Adapt your depth to the user's level."
        ),
        "welcome": "Hey! I'm your Data Science Expert. Ask me anything",
        "color": "#4A90D9",
    },
    "🌍 Travel Guide": {
        "system": (
            "You are an enthusiastic and knowledgeable travel guide with expertise in "
            "destinations worldwide. You give personalised recommendations for itineraries, "
            "local food, culture, hidden gems, visa tips, and packing advice. "
            "Be warm, inspiring, and practical."
        ),
        "welcome": "Ciao! I'm your Travel Guide. Where in the world are you dreaming of going?",
        "color": "#27AE60",
    },
    "🍕 Chef Assistant": {
        "system": (
            "You are a professional chef and culinary expert. You help users with recipes, "
            "ingredient substitutions, cooking techniques, meal planning, and kitchen tips. "
            "Be enthusiastic about food, suggest creative variations, and always provide "
            "clear step-by-step instructions when giving recipes."
        ),
        "welcome": "Welcome to the kitchen! I'm your Chef Assistant. What are we cooking today?",
        "color": "#E67E22",
    },
    "💪 Fitness Coach": {
        "system": (
            "You are a certified personal trainer and fitness coach. You provide workout plans, "
            "exercise techniques, nutrition advice, recovery tips, and motivation. "
            "Always ask about the user's fitness level and goals before giving advice. "
            "Prioritise safety and proper form in all recommendations."
        ),
        "welcome": "Let's get moving! I'm your Fitness Coach. What are your fitness goals?",
        "color": "#E74C3C",
    },
    "🏏 Cricket Expert": {
        "system": (
            "You are a cricket expert with encyclopaedic knowledge of cricket history, rules, "
            "statistics, players, tournaments, and tactics across all formats — Test, ODI, and T20. "
            "You can analyse matches, compare players across eras, explain techniques, and discuss "
            "team strategies. You're passionate about cricket and love a good debate."
        ),
        "welcome": "Howzat! I'm your Cricket Expert. Ask me anything — history, stats, tactics, players!",
        "color": "#8E44AD",
    },
}


# ── Session state defaults ─────────────────────────────────────────────────────
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖", layout="centered")

if st.session_state.selected_persona is None:
    st.title("🤖 LangChain Chatbot")
    st.markdown("### Choose your assistant to get started")
    st.markdown("---")
    
    cols = st.columns(2)
    persona_list = list(PERSONAS.keys())

    for i, name in enumerate(persona_list):    
        col = cols[i % 2]
        persona = PERSONAS[name]
        with col:
            st.markdown(
                f"""
                <div style="
                    border: 2px solid {persona['color']};
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 12px;
                    text-align: center;
                    height: 150px;
                ">
                    <h3 style="color: {persona['color']}; margin: 0;">{name}</h3>
                    <p style="font-size: 13px; color: gray; margin-top: 8px;">
                        {persona['welcome']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Chat with {name}", key=name, use_container_width=True):
                st.session_state.selected_persona = name
                st.session_state.messages = [
                    SystemMessage(content=persona["system"]),
                    AIMessage(content=persona["welcome"])
                ]
                st.rerun()


else:
    # Header
    persona = PERSONAS[st.session_state.selected_persona]
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            f"<h2 style='color: {persona['color']};'>{st.session_state.selected_persona}</h2>",
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("← Back"):
            st.session_state.selected_persona = None
            st.session_state.messages = []
            st.rerun()
 
    st.markdown("---")
 
    # Render chat history (skip SystemMessage, show the rest)
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            continue
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(msg.content)
 
    # User input
    if user_input := st.chat_input("Type your message..."):
        # Add and display user message
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)
 
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = model.invoke(st.session_state.messages,
                                        max_output_tokens=300)
                st.markdown(response.content)
 
        st.session_state.messages.append(AIMessage(content=response.content))
