from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("""
You are a cricket analyst.

Given the following information:

Player Name: {player_name}
Format: {format}
Country: {country}

Provide a detailed profile of the player including:

1. Full Name
2. Country
3. Playing Role (Batsman/Bowler/All-rounder/Wicketkeeper)
4. Batting Style
5. Bowling Style
6. Major Teams Played For
7. Career Statistics
8. Notable Achievements
9. Strengths
10. Recent Performance Summary
11. Interesting Facts

Return the response in well-structured markdown.

If the player is unknown, clearly state that the player could not be identified.
""")

template.save("02_prompts/player_profile_template.json")