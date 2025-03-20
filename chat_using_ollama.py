from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Redis, FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain,RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


def chat_simple_qa_3():
    #set_verbose(True)

    q3 = "Where is the Burj Khalifa located? "
    q3_template = """Question: {question} Answer: Be precise and answer in a single sentence."""

    prompt = PromptTemplate(template=q3_template, input_variables=["question"])
    ollam_llm = Ollama(model="mistral")
    llm_chain = LLMChain(prompt=prompt, llm=ollam_llm)

    print(llm_chain.invoke(q3))

def chat_using_rag():
    #doc = WebBaseLoader("https://www.bbc.com/news/uk-england-beds-bucks-herts-68539052").load()
    doc = TextLoader("iceland_volcano.txt").load()
    text_content_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    splitted_doc= text_content_splitter.split_documents(doc)

    vector_store = Redis.from_documents(splitted_doc,embedding=OllamaEmbeddings(),redis_url="redis://localhost:6379", index_name="stuff")
    #vector_store = FAISS.from_documents(splitted_doc, OllamaEmbeddings())
    
    vector_store_retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_llm(llm=Ollama(model="mistral"), retriever=vector_store_retriever)
    print("Ask any question regarding the iceland volcanic eruption.")
    # keep the bot running in a loop to simulate a conversation
    while True:
        question = input()
        result = qa({"query": question})
        print (result["result"])