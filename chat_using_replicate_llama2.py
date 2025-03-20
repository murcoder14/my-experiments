import os

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from llama_index import ServiceContext, VectorStoreIndex, LangchainEmbedding
from langchain_community.llms import Replicate
from langchain_community.vectorstores import Redis
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chat_via_hf_n_replicate():
    # Read the document using a DocumentLoader
    iceland_volcano_doc = TextLoader("iceland_volcano.txt").load()

    # We split the text content using the RecursiveTextSplitter which is recommended
    # for splitting generic text. See ,  https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter
    iceland_volcano_doc_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    splitted_doc = iceland_volcano_doc_splitter.split_documents(iceland_volcano_doc)

    # We instantiate the Embeddings Model we shall use
    embedder =  LangchainEmbedding(HuggingFaceEmbeddings())
    
    # set llm to be using Llama2 hosted on Replicate
    llama2_13b_chat = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"

    llm  = Replicate(
        model=llama2_13b_chat,
        model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
    )
    # create a ServiceContext instance to use Llama2 and custom embeddings
    llama2_hf_sc = ServiceContext.from_defaults(llm=llm, chunk_size=800, chunk_overlap=20, embed_model=embedder)

    # create vector store index from the documents created above
    index = VectorStoreIndex.from_documents(splitted_doc, service_context=llama2_hf_sc)

    # We then Ggnerate Embeddings for the text in the splitted document and store them in a local instance of Redis running on Docker.
    # It passes the splitted document to the OpenAPI embeddings model when doing this step.
    vector_store = Redis.from_documents(splitted_doc,embedding=embedder,redis_url="redis://localhost:6379", index_name="chunk")
    #results = rds.similarity_search("where does mrs ruan live")
    #print(results[0].page_content)

    #Redis.from_existing_index(embedding=embedder,redis_url="redis://localhost:6379", index_name="chunk").as_retriever()
    vector_store_retriever = vector_store.as_retriever()

    qa = RetrievalQA.from_llm(llm=ChatOpenAI(), retriever=vector_store_retriever)
    
    print("Ask any question regarding Iceland Volcano:")
    # keep the bot running in a loop to simulate a conversation
    while True:
        question = input()
        result = qa({"query": question})
        print (result["result"])