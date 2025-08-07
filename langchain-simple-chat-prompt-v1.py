import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()

MODEL=os.getenv("MODEL")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


prompt = ChatPromptTemplate([
    SystemMessage(content="You are teacher and you are expert to create questions for different level of grade."),
    MessagesPlaceholder("user_input")
])


model = init_chat_model(MODEL, model_provider="groq")
chain = prompt | model

question = "Give me 3 questions of physics for 5 grade"
input_data = {"user_input": [HumanMessage(content=question)]}
result = chain.invoke(input=input_data)
print(result.content)
