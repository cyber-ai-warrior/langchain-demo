import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

MODEL=os.getenv("MODEL")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


prompt = ChatPromptTemplate([
    ("system", "You are math teacher and you are expert to create questions for different level of grade."),
    ("user", "Give me {number_of_question} questions for grade: {grade}.")
])
model = init_chat_model(MODEL, model_provider="groq")
chain = prompt | model

number_of_question = 2
grade = "LKG"
input_data = {"number_of_question": number_of_question, "grade": grade}

result = chain.invoke(input=input_data)
print(result.content)
