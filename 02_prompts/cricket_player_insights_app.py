from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv()
import streamlit as st

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

template = load_prompt("02_prompts/player_profile_template.json")

chain = template | model

st.title("Cricket Player Insights App")

player_name = st.text_input("Enter the player's name:")
format = st.selectbox("Select the format:", ["Test", "ODI", "T20"])
country = st.text_input("Enter the player's country:")

result = chain.invoke({"player_name": player_name, "format": format, "country": country})

if st.button("Get Insights"):
    result = chain.invoke({"player_name": player_name, "format": format, "country": country})
    st.write(result.content)
