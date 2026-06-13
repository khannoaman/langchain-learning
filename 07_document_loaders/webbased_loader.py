from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field

load_dotenv()

class product(BaseModel):
    name: str = Field(description="Name of the product")
    original_price: str = Field(description="Original Price of the product")
    discounted_price: str = Field(description="Discounted Price of the product")
    rating: str = Field(description="Rating of the product")
    discount_percentage: str = Field(description="Discount Percentage of the product")
    brand: str = Field(description="Brand of the product")
    RAM: str = Field(description="RAM of the product")
    ROM: str = Field(description="ROM of the product")
    colour: str = Field(description="Colour of the product")
    processor: str = Field(description="Processor of the product")




prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a expert Name Entity Extractor from WebPages."),
        ("human", "Extract all the fields from the below web page and add NA if missing: {text} \n\n")
    ]
)

url ="https://www.flipkart.com/apple-macbook-air-m5-2026-m5-16-gb-512-gb-ssd-tahoe-mdhe4hn-a/p/itm8505e2f874525"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

loader = WebBaseLoader(url, header_template=headers)
docs = loader.load()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

structured_output = model.with_structured_output(product)

chain = prompt | structured_output

result = chain.invoke({"text": docs[0].page_content})

print(result)