from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

class CricketPlayerProfile(BaseModel):
    name: str = Field(description="The name of the cricket player")
    country: str = Field(description="The country the player represents")
    date_of_birth: str = Field(description="Date of birth in YYYY-MM-DD format")
    role: str = Field(description="Batsman, Bowler, Wicketkeeper, All-rounder")
    bowling_style: str = Field(description="Bowling style")
    batting_style: str = Field(description="Batting style")
    formats: list[str] = Field(description="Formats played: Test, ODI, T20")
    matches_played: dict[str, int] = Field(description="Matches played in each format")
    batting_statistics: dict[str, dict[str, float]] = Field(description="Batting stats by format")
    bowling_statistics: dict[str, dict[str, float]] = Field(description="Bowling stats by format")
    major_achievements: list[str] = Field(description="Major achievements and awards")

parser = PydanticOutputParser(pydantic_object=CricketPlayerProfile)

template = PromptTemplate(
    template="Give details about the following cricket player: {player}\n\n{format_instructions}",
    input_variables=["player"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

player = input("Enter the name of the cricket player: ")

result = chain.invoke({"player": player})

print(result)