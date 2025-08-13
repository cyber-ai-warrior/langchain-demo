import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()

MODEL=os.getenv("MODEL")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model = init_chat_model(MODEL, model_provider="groq")

@tool()
def multiply_my_number(a: int, b: int) -> int:
    """
    Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b

tool_model = model.bind_tools([multiply_my_number])

response = tool_model.invoke("how much 2 and 2?")

# Now, manually execute the tool call
tool_call = response.tool_calls[0]  # Get first tool call
tool_result = multiply_my_number.invoke(tool_call["args"])
# # Print the result
print(f"Tool result: {tool_result}")
