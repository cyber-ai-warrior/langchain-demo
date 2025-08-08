import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

MODEL=os.getenv("MODEL")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")


os.environ["GROQ_API_KEY"] = GROQ_API_KEY
model = init_chat_model(MODEL, model_provider="groq")

result = model.invoke("Five me 5 question of LLM")
print(result.content)
