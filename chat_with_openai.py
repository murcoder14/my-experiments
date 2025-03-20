from langchain.chains import LLMChain, RetrievalQA, create_retrieval_chain, create_history_aware_retriever
from langchain.chains.sequential import SimpleSequentialChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.globals import set_llm_cache
from langchain.globals import set_verbose
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from redis import Redis as RedisProto
from langchain_community.cache import InMemoryCache, RedisCache
from langchain_community.vectorstores import Redis
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# New Content Generation
def chat_simple_prompt_1():
    prompt_template = PromptTemplate.from_template("Tell me hot to use Generative AI for {topic}")
    model = OpenAI()
    output_parser = StrOutputParser()
    #print(model(prompt_template.format()))
    # LCEL illustrated below
    chain = prompt_template | model | output_parser
    print(chain.invoke({"topic":"HealthCare Gaps"}))

# Sentence or topic completion or elaboration
def chat_simple_prompt_2():
    # A prompt template with no variables or placeholders
    prompt_template = PromptTemplate.from_template("Controlling high blood pressure is important because...")
    model = OpenAI()
    print(model(prompt_template.format()))

# New Content Generation using a streaming response
def chat_simple_using_stream():
    prompt_template = PromptTemplate.from_template("Write a poem on the Moon.")
    model = OpenAI()
    for chunk in model.stream(prompt_template.format()):
        print(chunk,end='',flush=True)

# Content Generation that caches the responses in runtime memory and regurgitates it everytime
def qa_using_in_memory_cache():
    set_llm_cache(InMemoryCache())
    # A prompt template with placeholders
    prompt_template = PromptTemplate.from_template("Tell me a {adjective} {stuff} about {content}.")
    model = OpenAI()
    print(model(prompt_template.format(adjective="funny", stuff= "joke", content="cats")))
    print(model(prompt_template.format(adjective="funny",  stuff= "incident",content="London's weather")))
    print(model(prompt_template.format(adjective="funny", stuff= "joke", content="cats")))  # The joke about cats would be repeated due to the cache

# Content Generation that caches the responses in Redis cache and regurgitates it everytime
def qa_using_redis_cache():
    set_llm_cache(RedisCache(redis_=RedisProto()))
    prompt_template = PromptTemplate.from_template("Tell me a {adjective} joke about {content}.")
    model = OpenAI()
    print(model(prompt_template.format(adjective="funny", content="peppers")))
    print(model(prompt_template.format(adjective="funny", content="Ghost")))
    print(model(prompt_template.format(adjective="funny", content="peppers")))  # The joke for peppers would be repeated

# Conversational engagement to modify content
def chat_ask_help_1():
    set_verbose(True)
    chat_template = ChatPromptTemplate.from_messages(
        [   
            SystemMessage(content=("You are a helpful assistant that re-writes the user's text to sound more upbeat.")),
            HumanMessagePromptTemplate.from_template("{tm_message}"),
        ]
    )
    model = OpenAI()
    print(model(chat_template.format(tm_message="I don't like investing all my capital only in stocks.")))

def chat_ask_help_2():
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=("You shall respond only in JSON format.")),
            HumanMessagePromptTemplate.from_template("Top {n} countries in {region} by GDP"),
        ]
    )
    model = ChatOpenAI()
    print(model.invoke(chat_template.format(n='3',region="Asia")).content)

def chat_summarize_care_gaps():
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=("You shall respond only in a single sentence.")),
            #HumanMessagePromptTemplate.from_template("If Blood Pressure medication is not taken routinely then it can lead to health complications such as "),
            HumanMessagePromptTemplate.from_template("If Lead Screening in Children is not done regularly then exposure to lead can cause "),
        ]
    )
    model = ChatOpenAI()
    print(model.invoke(chat_template.format(n='3',region="Asia")).content)


# See the instructions to the LLM
def chat_simple_qa_1():
    #set_verbose(True)
    template = """Question: {question} Answer: Be precise """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI())
    question = input("Type your question: ")
    print(llm_chain.run(question))

# See the instructions to the LLM
def chat_simple_qa_2():
    #set_verbose(True)
    q2 = "What causes ischemic stroke? "
    q2_template = """Question: {question} Answer: Let's think step by step."""
    prompt = PromptTemplate(template=q2_template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI())
    print(llm_chain.run(q2))
    #print(llm_chain.invoke({"question": q2}))

def chat_multiple_qa_1():
    template = """Answer the following questions one at a time. Questions: {question} Answers: """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    model = OpenAI()
    llm_chain = LLMChain(prompt=prompt, llm=model)
    questions = [
        {'question': "Which team won the Bundesliga in the 2010-11 season?"},
        {'question': "Does Aspirin prevent death from a heart attack?"},
        {'question': "Who was the first person to reach the North Pole?"},
        {'question': "Where can I find a Walrus?"}
    ]
    print(llm_chain.generate(questions))

def chat_multiple_qa_2():
    template = """Answer the following questions one at a time. Questions: {questions} Answers: """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    model = OpenAI()
    llm_chain = LLMChain(prompt=prompt, llm=model)

    questions = (
    "Who won the men's French open tennis in 2019?\n" +
    "How far is New York City from Boston?\n" +
    "Who was the second person on the moon?" +
    "Can a polar bear live in the Arizona desert?"
    )
    print(llm_chain.run(questions))

def chat_using_context_1():
    # In the example below, the question is hardcoded inside the template definition. 
    
    template = """Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

    Context: Three years ago, when construction workers started demolishing a series of dams on the Hiitolanjoki River in Finland, they were greatly surprised to spot a run of salmon.
    The river used to be a key migration route for the endangered freshwater salmon from Lake Ladoga, in nearby Russia, to Finland. But between 1911 and 1925 the introduction of three dams 
    supplying hydroelectric energy created barriers between the salmon and their spawning grounds. The salmon and other fish, like brown trout, were trapped on the Finnish side of the river, 
    which remained fragmented for 100 years.

    Question: What surprised the construction workers?

    Answer: """

    model = OpenAI()
    print(model(template))
    
def chat_using_context_2():
    # In the example below, we supply the question at runtime which is then embedded within the template.
    
    template = """Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

    Context: Beet juice may help lower your blood pressure. Drinking beet juice may increase plasma nitrate levels and boost physical performance. Beets get their rich color from betalains, which are water-soluble antioxidants. 
    Betalains and other antioxidants may help find and destroy free radicals or unstable molecules in the body, which, in large numbers, can promote inflammation and increase the risk of cancer.
    Beets are a good source of potassium, a mineral and electrolyte that helps nerves and muscles function properly. Drinking beet juice in moderation can help keep your potassium levels optimal.

    Question: {query}

    Answer: """

    model = OpenAI()
    print(model(template.format(query="How is drinking Beet juice helpful?"))
    #print(model(template.format(query="Does drinking Beet juice reduce headache?"))
)

def generate_code_1():
    task_template1 = """You are an experienced C Programmer. Write a function in C to {task}"""
    prompt_template1 = PromptTemplate.from_template(template=task_template1)
    llm_chain_1 = LLMChain(prompt=prompt_template1, llm=ChatOpenAI(temperature=0.2))

    task_template2 = "Given the C {function}, describe it in detail"
    prompt_template2 = PromptTemplate.from_template(template=task_template2)
    llm_chain_2 = LLMChain(prompt=prompt_template2, llm=ChatOpenAI(temperature=1.2))

    linear_chain = SimpleSequentialChain(chains=[llm_chain_1,llm_chain_2])
    print(linear_chain.invoke("compute the first five Fibonacci numbers")["output"])

def simple_qa_using_rag():
    web_page = WebBaseLoader("https://www.nhlbi.nih.gov/health/atherosclerosis").load()
    splitted_doc = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20).split_documents(web_page)
    vector_store = Redis.from_documents(splitted_doc, embedding=OpenAIEmbeddings(), redis_url="redis://localhost:6379",index_name="guidei")

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}""")

    document_chain = create_stuff_documents_chain(ChatOpenAI(), prompt)
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), document_chain)

    # The retrieval method works on the most recent input,
    response = retrieval_chain.invoke({"input": "What is Renal artery stenosis?"})
    print(response["answer"])

def chat_using_rag():
    # Read the document using a DocumentLoader
    iceland_volcano_doc = TextLoader("iceland_volcano.txt").load()

    # We split the text content using the RecursiveTextSplitter which is recommended
    # for splitting generic text. See ,  https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter
    iceland_volcano_doc_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    splitted_doc = iceland_volcano_doc_splitter.split_documents(iceland_volcano_doc)

    # We instantiate the Embeddings Model we shall use
    embedder = OpenAIEmbeddings()

    # We then Generate Embeddings for the text in the splitted document and store them in a local instance of Redis running on Docker.
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


def chat_using_rag2():
    web_page = WebBaseLoader("https://my.clevelandclinic.org/health/diseases/16753-atherosclerosis-arterial-disease").load()
    splitted_doc = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20).split_documents(web_page)
    vector_store = Redis.from_documents(splitted_doc, embedding=OpenAIEmbeddings(), redis_url="redis://localhost:6379",index_name="guidei")

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retrieval_chain = create_history_aware_retriever(ChatOpenAI(),vector_store.as_retriever(),prompt)

    chat_history = [HumanMessage(content="Can atherosclerosis be prevented?"), AIMessage(content="Yes!")]
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    print(response)

# In the example below, we see a use case where we use RunnablePassthrough along with RunnableMap.
# RunnablePassthrough is useful for manipulating the output of one Runnable to match the input format of the next Runnable in a sequence.
def chat_demo_runnable_passthru():
    vectorstore = Redis.from_texts(["Five years ago Boeing faced one of the biggest scandals in its history, after two brand new 737 Max planes were lost in almost identical accidents that cost 346 lives."],
                                   embedding=OpenAIEmbeddings(), redis_url="redis://localhost:6379",index_name="boeings")
    retriever = vectorstore.as_retriever()
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Here the input to prompt is expected to be a map with keys “context” and “question”. The user input is just the question. So we need to get the context using our retriever and passthrough the user input
    # under the “question” key. In this case, the RunnablePassthrough allows us to pass on the user’s question to the prompt and model.
    retrieval_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | ChatOpenAI()
            | StrOutputParser()
    )
    print(retrieval_chain.invoke("What was the scandal Boeing faced?"))

# RunnableParallel (aka. RunnableMap) makes it easy to execute multiple Runnables in parallel and to return the output of these Runnables as a map.
def chat_demo_runnable_parallel():
    model = OpenAI()
    joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
    poem_chain = ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | model
    parallel_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)
    print(parallel_chain.invoke({"topic": "snow"}))

# RunnablePassthrough allows to pass inputs unchanged or with the addition of extra keys. This typically is used in conjuction with RunnableParallel to assign data to a new key in the map.
# RunnablePassthrough() called on it’s own, will simply take the input and pass it through.
# RunnablePassthrough called with assign (RunnablePassthrough.assign(...)) will take the input, and will add the extra arguments passed to the assign function.
#
def chat_demo_runnable_passthru_with_assign():
    runnable = RunnableParallel(
        set_verbose(True),
        given = RunnablePassthrough(),
        points = RunnablePassthrough.assign(weights = lambda x: x["honors_classes"] * 0.5),
        reported= RunnablePassthrough.assign(weighted_gpa = lambda y: y["unweighted_gpa"] + y["honors_classes"] * 0.5),
    )
    print(runnable.invoke({"unweighted_gpa": 4.06,"honors_classes":2}))
    # {'given': {'unweighted_gpa': 4.06, 'honors_classes': 2}, 'points': {'unweighted_gpa': 4.06, 'honors_classes': 2, 'weights': 1.0}, 'reported': {'unweighted_gpa': 4.06, 'honors_classes': 2, 'weighted_gpa': 5.06}}


def chat_demo_runnable_passthru_n_parallel():
    runnable = RunnableParallel(
        set_verbose(True),
        weighted_gpa  =   lambda y: y["unweighted_gpa"] + y["honors_classes"] * 0.5,
    )
    print(runnable.invoke({"unweighted_gpa": 4.06,"honors_classes":2}))
    # {'passed': {'age': 5}, 'extra': {'age': 5, 'mult': 15}, 'modified': 6}
