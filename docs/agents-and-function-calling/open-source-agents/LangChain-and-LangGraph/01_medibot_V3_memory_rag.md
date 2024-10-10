# Conversational Interface - Medical Clinic

Conversational interfaces such as chatbots and virtual assistants can be used to enhance the user experience for your customers. Chatbots uses natural language processing (NLP) and machine learning algorithms to understand and respond to user queries. Chatbots can be used in a variety of applications, such as customer service, sales, and e-commerce, to provide quick and efficient responses to users. They can be accessed through various channels such as websites, social media platforms, and messaging apps. In this notebook, we will build a chatbot using two popular Foundation Models (FMs) in Amazon Bedrock, Claude V3 Sonnet and Llama 3 8b.

## Set up: Introduction to ChatBedrock 

**Supports the following**
1. Multiple Models from Bedrock 
2. Converse API
3. Ability to do tool binding
4. Ability to plug with LangGraph flows

⚠️ ⚠️ ⚠️ Before running this notebook, ensure you've run the  set up libraries if you do not have the versions installed ⚠️ ⚠️ ⚠️


```python
# %pip install -U langchain-community>=0.2.12, langchain-core>=0.2.34
# %pip install -U --no-cache-dir  \
#     "langchain>=0.2.14" \
#     "faiss-cpu>=1.7,<2" \
#     "pypdf>=3.8,<4" \
#     "ipywidgets>=7,<8" \
#     matplotlib>=3.9.0 \
#     "langchain-aws>=0.1.17"
#%pip install -U --no-cache-dir boto3
#%pip install grandalf==3.1.2
```

### Set up classes

- helper methods to set up the boto 3 connection client which wil be used in any class used to connect to Bedrock
- this method accepts parameters like `region` and `service` and if you want to `assume any role` for the invocations
- if you set the  AWS credentials then it will use those


```python
import warnings

from io import StringIO
import sys
import textwrap
import os
from typing import Optional

# External Dependencies:
import boto3
from botocore.config import Config

warnings.filterwarnings('ignore')

def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        print(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        print("\n".join(textwrap.wrap(line, width=width)))
        

def get_boto_client_tmp_cred(
    retry_config = None,
    target_region: Optional[str] = None,
    runtime: Optional[bool] = True,
    service_name: Optional[str] = None,
):

    if not service_name:
        if runtime:
            service_name='bedrock-runtime'
        else:
            service_name='bedrock'

    bedrock_client = boto3.client(
        service_name=service_name,
        config=retry_config,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN',""),

    )
    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client    

def get_boto_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
    service_name: Optional[str] = None,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE", None)
    retry_config = Config(
        region_name=target_region,
        signature_version = 'v4',
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name
    else: # use temp credentials -- add to the client kwargs
        print(f"  Using temp credentials")

        return get_boto_client_tmp_cred(retry_config=retry_config,target_region=target_region, runtime=runtime, service_name=service_name)

    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if not service_name:
        if runtime:
            service_name='bedrock-runtime'
        else:
            service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client

```

### Boto3 client
- Create the run time client which we will use to run through the various classes


```python
#os.environ["AWS_PROFILE"] = '<replace with your profile if you have that set up>'
region_aws = 'us-east-1' #- replace with your region
boto3_bedrock = get_boto_client(region=region_aws, runtime=True, service_name='bedrock-runtime')
```

### LangChain Expression Language (LCEL):
According to LangChain: *"LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production)."*

On this tutorial we will be using **LangChain Expression Language** to define and invoke our Chatbots.

## Chatbot Architectures
Chatbots can come in many shape and sizes, all depending on its use-case. Some models are meant to return general information. Others might be catered to a particular audience thus its inferences might be curtained to a particular tone. And others might need relevant context to give out an informed response to the user. Most robust ones will draw from all architectures and build on it. Below are a few popular types of Chatbots.

1. **Chatbot (Naive)** - Zero-Shot chatbot with using FM model trained knowledge.
2. **Chatbot using prompt** - Template driven - Chatbot with some context provided in the prompt template.
3. **Chatbot with persona** - Chatbot with defined roles. i.e. Career Coach and Human interactions.
4. **Contextual-aware chatbot** - Passing in context through an external file by generating embeddings.

For this demo we will build a robust chatbot that will leverage an array of features drawn from the architectures above. But first lets dive deeper into the architectures.

#### Using `ChatBedrock` and `HumanMessage` objects to wrap up our message and invoke the LLM


```python
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_core.messages import HumanMessage

model_parameter = {"temperature": 0.0, "top_p": .5, "max_tokens_to_sample": 2000}
model_id = "meta.llama3-8b-instruct-v1:0"
bedrock_llm = ChatBedrock(
    model_id=model_id,
    client=boto3_bedrock,
    model_kwargs=model_parameter, 
    beta_use_converse_api=True
)

```



## Memory powered chatbot with Amazon Bedrock and LangChain
The previous chatbot was able to answer us without issues, however because it lacks memory or context is not able to be very useful.
In Conversational interfaces such as chatbots, it is highly important to remember previous interactions, both at a short term but also at a long term level.

LangChain provides memory components in two forms. First, LangChain provides helper utilities for managing and manipulating previous chat messages. These are designed to be modular and useful regardless of how they are used. Secondly, LangChain provides easy ways to incorporate these utilities into chains.
It allows us to easily define and interact with different types of abstractions, which make it easy to build powerful chatbots.

![Amazon Bedrock - Conversational Interface](./images/chatbot_bedrock.png)
*The UML above visualizes a conversational interface with Memory*

#### Simple Conversation chain 

**Uses the In memory Chat Message History**

The example below uses the same history for all sessions and shows how to self-manage chat history and later use _RunnableWithMessageHistory_ management which will allow us to also break down our chat history into multiple sessions.

**Note**
1. `Chat History` is a variable is a placeholder in the prompt template. which will have Human/Ai alternative messages
2. Human query is the final question as `Input` variable
3. `RunnableWithMessageHistory` is the class which we wrap the `chain` in to run with history. which is in [Docs link]('https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#')
4. Config object `{"configurable": {'session_id_variable':'value,....other keys}` is used by RunnableWithMessageHistory to manage multiple historical chat flows
5. Configuration gets passed in as invoke({dict}, config={"configurable": {"session_id": "abc123"}}) and it gets converted to `RunnableConfig` which is passed into every invoke method. To access this we need to extend the Runnable class and access it
6. The chain processes the inputs as a dict object


Wrap the rag_chain with RunnableWithMessageHistory to automatically handle chat history:

Any Chain wrapped with RunnableWithMessageHistory - will manage chat history variables appropriately, however the ChatTemplate should have the Placeholder for history


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt_with_history = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a pirate. Answer the following questions as best you can."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# Add history to the in-memory chat history
class ChatHistoryAdd(Runnable):
    def __init__(self, chat_history):
        self.chat_history = chat_history

    def invoke(self, _input: str, config: RunnableConfig = None) -> str:
        try:
            self.chat_history.add_ai_message(_input)
            return _input
        except Exception as e:
            return f"Error processing input: {str(e)}"
        
        
history = InMemoryChatMessageHistory()
chat_add = ChatHistoryAdd(history)

# second way to create a callback runnable function
def chat_user_input_add(input_dict: dict) -> dict:
    payload = f"\n{input_dict['input']}"
    history.add_user_message(payload) 
    return input_dict 

chat_user_add = RunnableLambda(chat_user_input_add)

memory_chain = (
    RunnablePassthrough() 
    | chat_user_add
    | prompt_with_history
    | bedrock_llm
    | chat_add
    | StrOutputParser()
)
print_ww(f"Chat history before invocation:\n{history}\n")
print_ww(f"Chat history before runnable invocation:\n{chat_add.chat_history}\n")
```

    Chat history before invocation:
    
    
    Chat history before runnable invocation:
    
    



```python
memory_chain.invoke(
    {"input": "what is the weather like in Seattle WA?", "chat_history": history.messages},
    config={"configurable": {"session_id": "abc123"}}
                    )
print_ww("\n-----------------------")
print_ww(f"Chat history after invocation:\n{history}")
```

    
    -----------------------
    Chat history after invocation:
    Human:
    what is the weather like in Seattle WA?
    AI:
    
    Arrr, shiver me timbers! As a pirate, I've had me share o' sailin' the seven seas, but I've also had
    me share o' dealin's with the scurvy dogs from Seattle, Washington. Now, about the weather in
    Seattle... (spits out a wad o' chewin' tobacco)
    
    Seattle's got a reputation fer bein' a damp and drizzly place, matey. The weather's often gray and
    overcast, with a misty rain that'll soak yer boots and make ye want to stay below deck. But don't ye
    worry, it's not all gloom and doom! The rain's not as fierce as the storms I've faced on the high
    seas, and the sun does peek out from behind the clouds every now and then.
    
    In the summer, the weather's a mite more pleasant, with temperatures in the mid-70s to mid-80s
    (that's 23 to 30 degrees Celsius fer ye landlubbers). But don't get too comfortable, or ye might
    find yerself caught in a sudden downpour!
    
    In the winter, it's a different story altogether. The rain's more frequent, and the temperatures can
    drop to the mid-30s to mid-40s (that's 2 to 7 degrees Celsius). But that's when the coffee flows
    like grog, and the locals gather 'round to swap tales o' the sea... er, I mean, the weather.
    
    So, if ye be plannin' a visit to Seattle, be sure to pack yer waterproof gear and a good sense o'
    humor. And if ye see me strollin' down the dock, don't be afraid to offer me a swig o' rum and a
    tale o' the sea!



```python
print_ww("\n-----------------------")
print_ww(f"Chat history using runnable invocation:\n{chat_add.chat_history}") 
```

    
    -----------------------
    Chat history using runnable invocation:
    Human:
    what is the weather like in Seattle WA?
    AI:
    
    Arrr, shiver me timbers! As a pirate, I've had me share o' sailin' the seven seas, but I've also had
    me share o' dealin's with the scurvy dogs from Seattle, Washington. Now, about the weather in
    Seattle... (spits out a wad o' chewin' tobacco)
    
    Seattle's got a reputation fer bein' a damp and drizzly place, matey. The weather's often gray and
    overcast, with a misty rain that'll soak yer boots and make ye want to stay below deck. But don't ye
    worry, it's not all gloom and doom! The rain's not as fierce as the storms I've faced on the high
    seas, and the sun does peek out from behind the clouds every now and then.
    
    In the summer, the weather's a mite more pleasant, with temperatures in the mid-70s to mid-80s
    (that's 23 to 30 degrees Celsius fer ye landlubbers). But don't get too comfortable, or ye might
    find yerself caught in a sudden downpour!
    
    In the winter, it's a different story altogether. The rain's more frequent, and the temperatures can
    drop to the mid-30s to mid-40s (that's 2 to 7 degrees Celsius). But that's when the coffee flows
    like grog, and the locals gather 'round to swap tales o' the sea... er, I mean, the weather.
    
    So, if ye be plannin' a visit to Seattle, be sure to pack yer waterproof gear and a good sense o'
    humor. And if ye see me strollin' down the dock, don't be afraid to offer me a swig o' rum and a
    tale o' the sea!



```python
# Follow-up question:
memory_chain.invoke(
    {"input": "What is its states capital?", "chat_history": history.messages}, 
    config={"configurable": {"session_id": "abc123"}})

print_ww("\n-----------------------")
print_ww(f"Chat history after second invocation:\n{history}")
```

    
    -----------------------
    Chat history after second invocation:
    Human:
    what is the weather like in Seattle WA?
    AI:
    
    Arrr, shiver me timbers! As a pirate, I've had me share o' sailin' the seven seas, but I've also had
    me share o' dealin's with the scurvy dogs from Seattle, Washington. Now, about the weather in
    Seattle... (spits out a wad o' chewin' tobacco)
    
    Seattle's got a reputation fer bein' a damp and drizzly place, matey. The weather's often gray and
    overcast, with a misty rain that'll soak yer boots and make ye want to stay below deck. But don't ye
    worry, it's not all gloom and doom! The rain's not as fierce as the storms I've faced on the high
    seas, and the sun does peek out from behind the clouds every now and then.
    
    In the summer, the weather's a mite more pleasant, with temperatures in the mid-70s to mid-80s
    (that's 23 to 30 degrees Celsius fer ye landlubbers). But don't get too comfortable, or ye might
    find yerself caught in a sudden downpour!
    
    In the winter, it's a different story altogether. The rain's more frequent, and the temperatures can
    drop to the mid-30s to mid-40s (that's 2 to 7 degrees Celsius). But that's when the coffee flows
    like grog, and the locals gather 'round to swap tales o' the sea... er, I mean, the weather.
    
    So, if ye be plannin' a visit to Seattle, be sure to pack yer waterproof gear and a good sense o'
    humor. And if ye see me strollin' down the dock, don't be afraid to offer me a swig o' rum and a
    tale o' the sea!
    Human:
    What is its states capital?
    AI:
    
    Arrr, ye want to know the capital o' Washington state, eh? Well, matey, I'll tell ye it's Olympia!
    That's right, Olympia be the capital o' Washington, and it's a fine place to visit, especially if ye
    be lookin' fer a taste o' politics and history. Just watch yer step, or ye might trip over a sea o'
    bureaucrats!



```python
chat_add.chat_history
```




    InMemoryChatMessageHistory(messages=[HumanMessage(content='\nwhat is the weather like in Seattle WA?'), AIMessage(content="\n\nArrr, shiver me timbers! As a pirate, I've had me share o' sailin' the seven seas, but I've also had me share o' dealin's with the scurvy dogs from Seattle, Washington. Now, about the weather in Seattle... (spits out a wad o' chewin' tobacco)\n\nSeattle's got a reputation fer bein' a damp and drizzly place, matey. The weather's often gray and overcast, with a misty rain that'll soak yer boots and make ye want to stay below deck. But don't ye worry, it's not all gloom and doom! The rain's not as fierce as the storms I've faced on the high seas, and the sun does peek out from behind the clouds every now and then.\n\nIn the summer, the weather's a mite more pleasant, with temperatures in the mid-70s to mid-80s (that's 23 to 30 degrees Celsius fer ye landlubbers). But don't get too comfortable, or ye might find yerself caught in a sudden downpour!\n\nIn the winter, it's a different story altogether. The rain's more frequent, and the temperatures can drop to the mid-30s to mid-40s (that's 2 to 7 degrees Celsius). But that's when the coffee flows like grog, and the locals gather 'round to swap tales o' the sea... er, I mean, the weather.\n\nSo, if ye be plannin' a visit to Seattle, be sure to pack yer waterproof gear and a good sense o' humor. And if ye see me strollin' down the dock, don't be afraid to offer me a swig o' rum and a tale o' the sea!", response_metadata={'ResponseMetadata': {'RequestId': '5c7ef488-d8db-436e-ab39-4fe9d7aa2e49', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Tue, 17 Sep 2024 14:23:50 GMT', 'content-type': 'application/json', 'content-length': '1552', 'connection': 'keep-alive', 'x-amzn-requestid': '5c7ef488-d8db-436e-ab39-4fe9d7aa2e49'}, 'RetryAttempts': 0}, 'stopReason': 'end_turn', 'metrics': {'latencyMs': 4776}}, id='run-55cf78ed-6bb3-4002-b321-6722daf97663-0', usage_metadata={'input_tokens': 46, 'output_tokens': 370, 'total_tokens': 416}), HumanMessage(content='\nWhat is its states capital?'), AIMessage(content="\n\nArrr, ye want to know the capital o' Washington state, eh? Well, matey, I'll tell ye it's Olympia! That's right, Olympia be the capital o' Washington, and it's a fine place to visit, especially if ye be lookin' fer a taste o' politics and history. Just watch yer step, or ye might trip over a sea o' bureaucrats!", response_metadata={'ResponseMetadata': {'RequestId': 'a28093de-ca89-406b-b611-c925b869767c', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Tue, 17 Sep 2024 14:24:00 GMT', 'content-type': 'application/json', 'content-length': '500', 'connection': 'keep-alive', 'x-amzn-requestid': 'a28093de-ca89-406b-b611-c925b869767c'}, 'RetryAttempts': 0}, 'stopReason': 'end_turn', 'metrics': {'latencyMs': 1140}}, id='run-36b70245-830e-4712-a6a5-9852fbf19d21-0', usage_metadata={'input_tokens': 427, 'output_tokens': 84, 'total_tokens': 511})])



## Building Chatbot with Context

There are many ways to give our chatbot context. One of the most effective approaches is giving relevant context to our chatbot by retrieving only those relevant pieces to answer the question using vectors, highly-dimensional data generated by an `embeddings model` (algorithms trained to encapsulate information into dense representations in a multidimensional space) and stored in a Vector Database. First we **generate embeddings** by passing our available context to our embeddings model, typically, you will have an ingestion process which will run through your embedding model and generate the embeddings and store them. In this example we are using our own Titan Embeddings model.

First step is to chunk our available context, embedd each chunk using our embeddings model and finally store all the chunks into a Vector Database or Vector Store for further retrival.

![Embeddings](./images/embeddings_lang.png)

Second process is the user request orchestration. User interaction with the chatbot, document retrieval, and finally invoking and returning the answer.

![Chatbot](./images/chatbot_lang.png)

### Architecture of a context powered Chatbot
![4](./images/context-aware-chatbot.png)

### FAISS as VectorStore

In order to be able to use embeddings for search, we need a store that can efficiently perform vector similarity searches. In this notebook we use FAISS, which is an in memory store. For permanently store vectors, one can use pgVector, Pinecone or Chroma.

The langchain VectorStore API's are available [here](https://python.langchain.com/en/harrison-docs-refactor-3-24/reference/modules/vectorstore.html)

To know more about the FAISS vector store please refer to this [document](https://arxiv.org/pdf/1702.08734.pdf).


#### Titan embeddings Model

Embeddings are a way to represent words, phrases or any other discrete items as vectors in a continuous vector space. This allows machine learning models to perform mathematical operations on these representations and capture semantic relationships between them.

Embeddings are for example used for the RAG [document search capability](https://labelbox.com/blog/how-vector-similarity-search-works/) 


```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings

br_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)

loader = CSVLoader("./rag_data/medi_history.csv")
documents_aws = loader.load() #
print(f"Number of documents={len(documents_aws)}")

docs = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separator=",").split_documents(documents_aws)

print(f"Number of documents after split and chunking={len(docs)}")
vectorstore_faiss_aws = FAISS.from_documents(
    documents=docs,
     embedding = br_embeddings
)
print(f"vectorstore_faiss_aws: number of elements in the index={vectorstore_faiss_aws.index.ntotal}::")
```

    Number of documents=7
    Number of documents after split and chunking=7
    vectorstore_faiss_aws: number of elements in the index=7::



```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

condense_question_system_template = (
    """
    You are an assistant for question-answering tasks. ONLY Use the following pieces of retrieved context to answer the question.
    If the answer is not in the context below , just say you do not have enough context. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Context: {context} 
    """
)

condense_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_question_system_template),
        ("human", "{input}"), 
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def debug_inputs(input_dict: dict) -> dict:
    return input_dict 

chat_user_debug = RunnableLambda(debug_inputs)

# The chain 
qa_chain = (
    {
        "context": vectorstore_faiss_aws.as_retriever() | format_docs, # can work even without the format
        "input": RunnablePassthrough(),
    }
    | chat_user_debug
    | condense_question_prompt
    | bedrock_llm
    | StrOutputParser()
)

print_ww(qa_chain.invoke(input="What are autonomous agents?")) 

print_ww(qa_chain.invoke(input="What all pain medications can be used for headache?")) 
```

    
    
    I do not have enough context to answer this question.
    
    
    According to the context, Aspirin can be used primarily for headache. Additionally, with Aspirin,
    you can generally take Ibruphen and Tylenol.


### Conversation Role driven Chatbot with History and a Retriever.
Wrap with Runnable Chat History and run the chat conversation

![Amazon Bedrock - Conversational Interface](./images/context_aware_history_retriever.png)

borrowed from https://github.com/langchain-ai/langchain





```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

### This below LEVERAGES the In-memory with multiple sessions and session id
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


contextualized_question_system_template = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualized_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualized_question_system_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If the answer is not present in the context, just say you do not have enough context to answer. \
If the input is not present in the context, just say you do not have enough context to answer. \
If the question is not present in the context, just say you do not have enough context to answer. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(bedrock_llm, qa_prompt)

rag_chain = create_retrieval_chain(vectorstore_faiss_aws.as_retriever(), question_answer_chain) 
chain_with_history_and_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

```


```python
result = chain_with_history_and_rag.invoke(
    {"input": "What all pain medications can be used for headache?"},
    config={"configurable": {"session_id": "session_1"}}
)
result
```




    {'input': 'What all pain medications can be used for headache?',
     'chat_history': [],
     'context': [Document(metadata={'source': './rag_data/medi_history.csv', 'row': 6}, page_content='What all pain medications can be used for headache?: what all pain medications can be used?\nFor your use case only Asprin can be used: Asprin can be used primarily'),
      Document(metadata={'source': './rag_data/medi_history.csv', 'row': 5}, page_content='What all pain medications can be used for headache?: what all pain killers can be used?\nFor your use case only Asprin can be used: Asprin can be used primarily'),
      Document(metadata={'source': './rag_data/medi_history.csv', 'row': 1}, page_content='What all pain medications can be used for headache?: What pain medications can be used Asprin?\nFor your use case only Asprin can be used: With Asprin you can generally take ibruphen, tylenol'),
      Document(metadata={'source': './rag_data/medi_history.csv', 'row': 3}, page_content='What all pain medications can be used for headache?: what types of pain can be treated with asprin?\nFor your use case only Asprin can be used: Asprin can be used to treat headache, body pain')],
     'answer': '\n\nAccording to the context, Aspirin can be used primarily for headache treatment. Additionally, it is mentioned that with Aspirin, you can generally take Ibruphen and Tylenol for headache treatment.'}




```python
result['chat_history']
```




    []





### As a follow on question

1. The phrase `it` will be converted based on the chat history
2. Retriever gets invoked to get relevant content based on chat history 


```python
follow_up_result = chain_with_history_and_rag.invoke(
    {"input": "What are medicines does it interfere with?"},
    config={"configurable": {"session_id": "session_1"}}
)
print_ww(follow_up_result)
```

    {'input': 'What are medicines does it interfere with?', 'chat_history': [HumanMessage(content='What
    all pain medications can be used for headache?'), AIMessage(content='\n\nAccording to the context,
    Aspirin can be used primarily for headache treatment. Additionally, it is mentioned that with
    Aspirin, you can generally take Ibruphen and Tylenol for headache treatment.')], 'context':
    [Document(metadata={'source': './rag_data/medi_history.csv', 'row': 2}, page_content='What all pain
    medications can be used for headache?: what pain medications does Asprin interfere with?\nFor your
    use case only Asprin can be used: With Asprin you can generally take all medicines except for XYZ'),
    Document(metadata={'source': './rag_data/medi_history.csv', 'row': 5}, page_content='What all pain
    medications can be used for headache?: what all pain killers can be used?\nFor your use case only
    Asprin can be used: Asprin can be used primarily'), Document(metadata={'source':
    './rag_data/medi_history.csv', 'row': 0}, page_content='What all pain medications can be used for
    headache?: what is asprin used for?\nFor your use case only Asprin can be used: Asprin is used for
    treating headache issues, pain  and also for thinning blood'), Document(metadata={'source':
    './rag_data/medi_history.csv', 'row': 6}, page_content='What all pain medications can be used for
    headache?: what all pain medications can be used?\nFor your use case only Asprin can be used: Asprin
    can be used primarily')], 'answer': '\n\nAccording to the context, Aspirin interferes with medicines
    except for XYZ.'}



```python
follow_up_result = chain_with_history.invoke(
    {"input": "Will it help with pain?"},
    config={"configurable": {"session_id": "session_1"}}
)
print_ww(follow_up_result)
```

    {'input': 'Will it help with pain?', 'chat_history': [HumanMessage(content='What all pain
    medications can be used for headache?'), AIMessage(content='\n\nAccording to the context, Aspirin
    can be used primarily for headache treatment. Additionally, it is mentioned that with Aspirin, you
    can generally take Ibruphen and Tylenol for headache treatment.'), HumanMessage(content='What are
    medicines does it interfere with?'), AIMessage(content='\n\nAccording to the context, Aspirin
    interferes with medicines except for XYZ.')], 'context': [Document(metadata={'source':
    './rag_data/medi_history.csv', 'row': 4}, page_content='What all pain medications can be used for
    headache?: what muscle pain can be trated with asprin?\nFor your use case only Asprin can be used:
    Asprin can be used to treat all types of muscle pain'), Document(metadata={'source':
    './rag_data/medi_history.csv', 'row': 5}, page_content='What all pain medications can be used for
    headache?: what all pain killers can be used?\nFor your use case only Asprin can be used: Asprin can
    be used primarily'), Document(metadata={'source': './rag_data/medi_history.csv', 'row': 6},
    page_content='What all pain medications can be used for headache?: what all pain medications can be
    used?\nFor your use case only Asprin can be used: Asprin can be used primarily'),
    Document(metadata={'source': './rag_data/medi_history.csv', 'row': 0}, page_content='What all pain
    medications can be used for headache?: what is asprin used for?\nFor your use case only Asprin can
    be used: Asprin is used for treating headache issues, pain  and also for thinning blood')],
    'answer': '\n\nAccording to the context, Aspirin can be used to treat all types of muscle pain.'}



```python

```
