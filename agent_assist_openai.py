from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_openai_tools_agent,AgentExecutor
from langchain.chains import LLMMathChain
from langchain_core.tools import Tool

def assist_me():

    """
        In general, connection strings have the form "dialect+driver://username:password@host:port/database"
        There are three components to the connection string in this exercise: 
        
        the dialect and driver      ('postgresql+psycopg2://'),                                         followed by 
        the username and password   ('student:datacamp'),                                               followed by 
        the host and port           ('@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com:5432/'),     and finally, 
        the database name           ('census').
    """

    # A percent symbol in the password should be encoded using the URL encoded string value, https://www.urlencoder.org/
    db = SQLDatabase.from_uri("postgresql+psycopg2://tm6492:ganeSHA24@localhost:5432/experiments")
    print(db.dialect)
    print(db.get_usable_table_names())
    print(db.run("SELECT * FROM products;"))


    QUERY = """
    Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
    Use the following format:

    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here.

    {question}
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    #print(agent_executor.invoke("List the number of tables in the schema")["output"])
    #print(agent_executor.invoke("List the tables with the largest number of rows")["output"])
    # print(agent_executor.invoke("Describe the table books")["output"])
    #print(agent_executor.invoke("How many products are there?")["output"])
    #print(agent_executor.invoke("List the names of all the products")["output"])
    #print(agent_executor.invoke("List the names of all the books")["output"])
    #print(agent_executor.invoke("List the name of the most expensive book")["output"])
    #print(agent_executor.invoke("Which is the least expensive drink")["output"])

    # It is unable to answer these correctly because of the lack of a Math tool. Need to determine how to enable a Math tool
    print(agent_executor.invoke("If I purchased 2 espresso and 1 Latte, what is the total price I paid for the products")["output"])
    #print(agent_executor.invoke("How much did Murali pay if he purchased a copy of the books authored by Nadella and Strang?")["output"])

def do_calc():
    model = ChatOpenAI(temperature=0.1)
    llm_math = LLMMathChain.from_llm(llm=model)

    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", 
             """You are a mathematical assistant. Use your tools to answer questions. If you do not have a tool to answer the question, say so.  Return only the answers. e.g Human: What is 12 + 7? AI: 19"""),
             MessagesPlaceholder("chat_history", optional=True),
             ("human", "{input}"),
             MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # initialize the math tool and include it in a list
    math_tool = Tool(name='Calculator',func=llm_math.run, description='Useful to answer questions about math.')
    toolkit = [math_tool]
    
    # Construct the OpenAI Tools agent
    agent = create_openai_tools_agent(model,toolkit, chat_prompt_template)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

    #print(agent_executor.invoke({"input": "What is 15 times 7?"}))
    #print(agent_executor.invoke({"input": "What is (4.5*2.0)^3.0?"}))
    #print(agent_executor.invoke({"input": "If there are 7 ways to pick a sandwich, 5 ways to pick a side and 4 ways to pick a drink, how many lunch combinations are possible?"}))
    print(agent_executor.invoke({"input": "If Amanda has four apples and Kelly brings two and a half apple boxes (apple box contains eight apples), how many apples do we have?"}))

