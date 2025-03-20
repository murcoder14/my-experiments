

from dotenv import load_dotenv
from agent_using_langgraph_01 import forecast_weather
from agent_assist_openai import assist_me,do_calc
from use_openai_n_agents import do_math
import chat_with_openai as oai
import chat_using_hf_llama2 as hf
import chat_using_ollama as ollama
import chat_with_anthropic as anth

import argparse

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--task",default="Print Welcome to Langchain")
    parser.add_argument("--language",default="python")
    args = parser.parse_args()

    # AGENTs
    #do_it()
    #do_it_again()
    #forecast_weather()
    #assist_me()
    #do_calc()
    #do_math()

    # ******************** HUGGING FACE **********************************
    #hf.chat_using_rag()

    # ******************** OLLAMA **********************************
    #ollama.chat_simple_qa_3()
    #ollama.chat_using_rag()

    #******************** OPENAI **********************************

    oai.chat_summarize_care_gaps()

    #oai.chat_simple_prompt_1()
    #oai.chat_simple_prompt_2()
    #oai.chat_simple_using_stream()
    #oai.chat_ask_help_1()
    #oai.chat_ask_help_2()
    #oai.qa_using_in_memory_cache()
    #oai.qa_using_redis_cache()
    #oai.chat_simple_qa_1()
    #oai.chat_simple_qa_2()
    #oai.chat_multiple_qa_1()
    #oai.chat_multiple_qa_2()
    #oai.chat_using_context_1()
    #oai.chat_using_context_2()
    #oai.simple_qa_using_rag()
    #oai.chat_using_rag()
    #oai.chat_using_rag2()
    #oai.generate_code_1()
    #oai.chat_demo_runnable()
    #oai.chat_demo_runnable_parallel()
    #oai.chat_demo_runnable_passthru_with_assign()
    #oai.chat_demo_runnable_passthru_n_parallel()

    # ******************** ANTHROPIC **********************************

    #anth.simple_chat_4()
    #anth.chat_using_rag()

if __name__ == "__main__":
    main()