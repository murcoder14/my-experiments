from typing import Dict, TypedDict, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph
from langchain_community.utilities import OpenWeatherMapAPIWrapper

class GraphState(TypedDict):
    question: Optional[str] = None
    classification: Optional[str] = None
    response: Optional[str] = None

def f1(input_1):
    return input_1 + ", Good "

def f2(input_2):
    return input_2 + " Day!"

def f1A(input_1):
    model = ChatOpenAI(temperature=0)
    response = model.invoke(input_1)
    return response.content

def f2A(input_2):
    return "Agent says, " + input_2

def f3(input_3):
    complete_query = "Your task is to provide only the city name based on the user query. \
        Nothing more, just the city name mentioned. Following is the user query: " + input_3
    model = ChatOpenAI(temperature=0) 
    response = model.invoke(complete_query)
    return response.content

def f4(input_4):
    weather = OpenWeatherMapAPIWrapper()
    weather_data = weather.run(input_4)
    return weather_data

def do_it():
    workflow = Graph()
    workflow.add_node("node_1",f1)
    workflow.add_node("node_2",f2)
    workflow.add_edge("node_1","node_2")
    workflow.set_entry_point("node_1")
    workflow.set_finish_point("node_2")
    app = workflow.compile()

    print(app.invoke("Murali"))

def do_it_again():
    workflow = Graph()
    workflow.add_node("agent",f1A)
    workflow.add_node("node_2A",f2A)
    workflow.add_edge("agent","node_2A")
    workflow.set_entry_point("agent")
    workflow.set_finish_point("node_2A")
    app = workflow.compile()

    print(app.invoke("Hey there!"))

def forecast_weather():
    workflow = Graph()
    workflow.add_node("agent",f3)
    workflow.add_node("node",f4)
    workflow.add_edge("agent","node")
    workflow.set_entry_point("agent")
    workflow.set_finish_point("node")
    app = workflow.compile()

    print(app.invoke("What is the temperature in Milan, Italy?"))