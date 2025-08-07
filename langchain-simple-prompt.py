import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

load_dotenv()

MODEL=os.getenv("MODEL")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


formal_template = "Five me {number_of_question} question on topic: {topic}"
prompt = PromptTemplate(template=formal_template, input_variables=["number_of_question", "topic"])
model = init_chat_model(MODEL, model_provider="groq")
chain = prompt | model

number_of_question = 5
topic = "lenskard"
input_data = {"number_of_question": number_of_question, "topic": topic}

result = chain.invoke(input=input_data)
print(result.content)
