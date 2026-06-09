from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = JsonOutputParser()

prompt1 = PromptTemplate(template="Give the statistics of the cricket player {player} in IPL format.\n\n {format_instructions}", input_variables=["player"], partial_variables={"format_instructions": parser.get_format_instructions()})

prompt2 = PromptTemplate(template="Give the statistics of the cricket player {player} in T20 International format.\n\n {format_instructions}", input_variables=["player"], partial_variables={"format_instructions": parser.get_format_instructions()})

chain = RunnableParallel({
    "ipl_stats": prompt1 | model | parser,
    "t20i_stats": prompt2 | model | parser
})

player_name = input("Enter the player's name: ")
result = chain.invoke({"player": player_name})

print(result)