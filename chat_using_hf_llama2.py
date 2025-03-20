from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.globals import set_verbose
from langchain_community.document_loaders import TextLoader
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.vectorstores import Redis
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chat_using_rag():
    set_verbose(True)

    iceland_volcano_doc = TextLoader("iceland_volcano.txt").load()
    iceland_volcano_doc_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    splitted_doc = iceland_volcano_doc_splitter.split_documents(iceland_volcano_doc)

    #embedder =  HuggingFaceInferenceAPIEmbeddings(api_key="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")
    
    hf = HuggingFaceEndpoint(endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
                             huggingfacehub_api_token="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")

    vector_store = Redis.from_documents(splitted_doc,embedding=OpenAIEmbeddings(),redis_url="redis://localhost:6379", index_name="volcanoes")
    qa = RetrievalQA.from_llm(llm=hf, retriever=vector_store.as_retriever())
    
    print("Ask any question regarding Iceland Volcano:")
    # keep the bot running in a loop to simulate a conversation
    while True:
        question = input()
        result = qa({"query": question})
        print (result["result"])