import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

MODEL=os.getenv("MODEL")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model = init_chat_model(MODEL, model_provider="groq")
grade5_chain = ChatPromptTemplate.from_template("Give me 5 question for grade 5 on topic: {topic}") | model
grade6_chain = (
    ChatPromptTemplate.from_template("Give me 5 question for grade 6 on topic: {topic}") | model
)

map_chain = RunnableParallel(grade_5_chain=grade5_chain, grade_6_chain=grade6_chain)

result = map_chain.invoke({"topic": "science"})
print(result.get('grade_5_chain').content)
print("-" * 100)
print(result.get('grade_6_chain').content)
