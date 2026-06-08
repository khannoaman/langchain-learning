from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
)

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
model2 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a detailed report on the {topic}.",
    input_variables=["topic"]
)   
prompt2 = PromptTemplate(
    template = "Prepare quick notes summarizing the following report:\n\n{report}",
    input_variables=["report"]
)

prompt3 = PromptTemplate(
    template = "Prepare 5 MCQs based on the following report:\n\n{report}",
    input_variables=["report"]
)

prompt4 = PromptTemplate(
    template = "merge the following notes and MCQs into a single document:\n\nNotes:\n{notes}\n\nMCQs:\n{mcqs}",
    input_variables=["notes", "mcqs"]   
)

parallel_chain = RunnableParallel(
    {
        "notes": prompt2 | model1 | parser,
        "mcqs": prompt3 | model1 | parser
    }
)

merge_chain = prompt4 | model1 | parser

chain = prompt1 | model1 | parser | parallel_chain | merge_chain

topic = input("Enter a topic to generate a report: ")

result = chain.invoke({"topic": topic})

print(result)


chain.get_graph().print_ascii()