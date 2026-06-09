from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableLambda, RunnableBranch


load_dotenv()

class Player(BaseModel):
    role : Literal["Batsman", "Bowler", "Wicketkeeper", "All-rounder"] = Field(description="Role of the player in the match: Batsman, Bowler, Wicketkeeper, All-rounder")
    name: str = Field(description="The name of the cricket player")
    country: str = Field(description="The country the player represents")

parser = PydanticOutputParser(pydantic_object=Player)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt1 = PromptTemplate(template="Give details about the following cricket player: {player}\n\n{format_instructions}", input_variables=["player"], partial_variables={"format_instructions": parser.get_format_instructions()})

role_chain = prompt1 | model | parser | RunnableLambda(lambda inputs: {"role": inputs.role, "player": inputs.name, "country": inputs.country})


json_parser = JsonOutputParser()

prompt2 = PromptTemplate(template="Give the batting statistics of the following player: {player}, {country}\n\n{format_instructions}", input_variable = ["player","country"],partial_variables={"format_instructions": json_parser.get_format_instructions()})

prompt3 = PromptTemplate(template="Give the bowling statistics of the following player: {player}, {country}\n\n{format_instructions}", input_variable = ["player","country"],partial_variables={"format_instructions": json_parser.get_format_instructions()})

prompt4 = PromptTemplate(template="Give the batting and bowling statistics of the following allrounder player: {player}, {country}\n\n{format_instructions}", input_variable = ["player","country"],partial_variables={"format_instructions": json_parser.get_format_instructions()})

prompt5 = PromptTemplate(template="Give the batting and wicketkeeper statistics of the following player: {player}, {country}\n\n{format_instructions}", input_variable = ["player","country"],partial_variables={"format_instructions": json_parser.get_format_instructions()})



branch = RunnableBranch(
    (lambda inputs: inputs["role"] == "Batsman", prompt2 | model | json_parser),
    (lambda inputs: inputs["role"] == "Bowler", prompt3 | model | json_parser),
    (lambda inputs: inputs["role"] == "All-rounder", prompt4 | model | json_parser),
    (lambda inputs: inputs["role"] == "Wicketkeeper", prompt5 | model | json_parser),
    RunnableLambda(lambda inputs: "Invalid Role provided. Please enter either 'Batsman', 'Bowler', 'Wicketkeeper', or 'All-rounder'.")
)

chain = role_chain | branch


player_name = input("Enter the name of the cricket player: ")

result = chain.invoke({"player": player_name})

print(result)

chain.get_graph().print_ascii()