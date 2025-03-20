from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# setup the tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def square(a) -> int:
    """Calculate the square of a number."""
    a = int(a)
    return a * a

def do_math(): 
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             """You are a mathematical assistant. Use your tools to answer questions. If you do not have a tool to answer the question, say so.  Return only the answers. e.g Human: What is 1 + 1? AI: 2"""),
             MessagesPlaceholder("chat_history", optional=True),
             ("human", "{input}"),
             MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Choose the LLM that will drive the agent
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    # setup the toolkit
    toolkit = [add, multiply, square]

    # Construct the OpenAI Tools agent
    agent = create_openai_tools_agent(llm, toolkit, prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

    #result = agent_executor.invoke({"input": "what is the sum of 24 and 71?"})
    #result = agent_executor.invoke({"input": "what is the 17 multiplied by 9?"})
    result = agent_executor.invoke({"input": "what is the square of 18?"})

    print(result['output'])