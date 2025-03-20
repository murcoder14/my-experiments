from langchain_community.document_loaders import TextLoader
from langchain.chains import LLMChain,RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Redis
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def simple_chat_translate():
    chat = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language}."
    )
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat
    print(chain.invoke({
        "input_language": "English",
        "output_language": "Malayalam",
        "text": "I love Rust",
    })
)

def chat_using_rag():
    doc = TextLoader("iceland_volcano.txt").load()
    text_content_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    splitted_doc = text_content_splitter.split_documents(doc)

    # Anthropic doesn't offers its own Embeddings model. They recommend using an Embedding model from Voyage AI but we use the OpenAIEmbeddings below.
    vector_store = Redis.from_documents(splitted_doc, embedding=OpenAIEmbeddings(), redis_url="redis://localhost:6379",index_name="volcanoes")
    vector_store_retriever = vector_store.as_retriever()

    qa = RetrievalQA.from_llm(llm=ChatAnthropic(model="claude-3-opus-20240229"), retriever=vector_store_retriever)
    print("Ask any question regarding the iceland volcanic eruption.")
    while True:
        question = input()
        result = qa({"query": question})
        print(result["result"])