from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

#schema for structured output
json_schema = {
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of key themes in the review"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Overall sentiment of the review"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "List of pros"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "List of cons"
        },
        "name_of_reviewer": {
            "type": ["string", "null"],
            "description": "Name of the reviewer (who reviewed) if mentioned in the review"
        },
        "ratings": {
            "type": ["number", "null"],
            "description": "Rating given by the reviewer if mentioned in the review"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_output = model.with_structured_output(json_schema)

review = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Khan
Ratings: 4.3/5"""

result = structured_output.invoke(review)

print(result)