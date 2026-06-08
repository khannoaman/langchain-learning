from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

load_dotenv()

class Review(BaseModel):
    review: str = Field(description="The generated review for the product")

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")

class Response(BaseModel):
    response: str = Field(description="The response to the customer based on the review sentiment") 

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                            max_output_tokens=200)

parser = StrOutputParser()

structured_output1 = model.with_structured_output(Review)
structured_output2 = model.with_structured_output(Sentiment)
structured_output3 = model.with_structured_output(Response)



prompt1 = PromptTemplate(
    template="Write a short {sentiment} review on the given product {product}.",
    input_variables=["sentiment", "product"]
)

prompt2 = PromptTemplate(
    template="Predict the sentiment of the following review:\n\n{review}",
    input_variables=["review"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to customer for the following positive review:\n\n{review}",
    input_variables=["review"]
)


prompt4 = PromptTemplate(
    template="Write an appropriate response to customer for the following negative review:\n\n{review}",
    input_variables=["review"]
)


product = input("Enter a product name: ")
sentiment = input("Enter the sentiment for the review (positive/negative): ")



branch = RunnableBranch(
    (lambda inputs: inputs.sentiment == "positive", prompt3 | structured_output3 ), 
    (lambda inputs: inputs.sentiment == "negative", prompt4 | structured_output3 ),
    RunnableLambda(lambda inputs: "Invalid sentiment provided. Please enter either 'positive' or 'negative'.")
)


chain = prompt1 | structured_output1 | RunnablePassthrough(lambda x: x.review) | prompt2 | structured_output2 | branch

result = chain.invoke({"product": product, "sentiment": sentiment})

print(result.response)

chain.get_graph().print_ascii()