import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
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

def _check_question_format(inputs):
    if not isinstance(inputs, list):
        return "Output format is invalid."

    for input in inputs:
        if not isinstance(input, dict) or "number" not in input or "question" not in input:
            return "Question number format is invalid."
    return inputs

parser = JsonOutputParser()

model = init_chat_model(MODEL, model_provider="groq")
chain = prompt | model | parser | _check_question_format

question = ("Give me 3 questions of physics for 5 grade. "
            "In the format of json list of questions and each question have key 'number' and 'question'. "
            "Output must be a list. nothing else")
input_data = {"user_input": [HumanMessage(content=question)]}
result = chain.invoke(input_data)
print(result)