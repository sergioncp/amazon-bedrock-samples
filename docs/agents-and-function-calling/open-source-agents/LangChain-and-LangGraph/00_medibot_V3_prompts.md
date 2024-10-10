# Conversational Interface - Medical Clinic

Conversational interfaces such as chatbots and virtual assistants can be used to enhance the user experience for your customers. Chatbots uses natural language processing (NLP) and machine learning algorithms to understand and respond to user queries. Chatbots can be used in a variety of applications, such as customer service, sales, and e-commerce, to provide quick and efficient responses to users. They can be accessed through various channels such as websites, social media platforms, and messaging apps. In this notebook, we will build a chatbot using two popular Foundation Models (FMs) in Amazon Bedrock, Claude V3 Sonnet and Llama 3 8b.


## Chatbot using Amazon Bedrock

![Amazon Bedrock - Conversational Interface](./images/chatbot_bedrock.png)


## Set up: Introduction to ChatBedrock and prompt templates

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

### Test the connectivity to Bedrock service


```python
import boto3
print(boto3.__version__)

#os.environ["AWS_PROFILE"] = '<replace with your profile if you have that set up>'
region_aws = 'us-east-1' #- replace with your region
boto3_client =  get_boto_client(region=region_aws, runtime=False, service_name='bedrock') # switch to the region of your choice
boto3_client.list_foundation_models()


```

### Boto3 client
- Create the run time client which we will use to run through the various classes


```python
#os.environ["AWS_PROFILE"] = '<replace with your profile if you have that set up>'
region_aws = 'us-east-1' #- replace with your region
boto3_bedrock = get_boto_client(region=region_aws, runtime=True, service_name='bedrock-runtime')
```

    Create new client
      Using region: us-east-1
      Using temp credentials
    boto3 Bedrock client successfully created!
    bedrock-runtime(https://bedrock-runtime.us-east-1.amazonaws.com)


### LangChain Expression Language (LCEL):
According to LangChain: *"LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production)."*

On this tutorial we will be using **LangChain Expression Language** to define and invoke our Chatbots.

## Chatbot Architectures
Chatbots can come in many shape and sizes, all depending on its use-case. Some models are meant to return general information. Others might be catered to a particular audience thus its inferences might be curtained to a particular tone. And others might need relevant context to give out an informed response to the user. Most robust ones will draw from all architectures and build on it. Below are a few popular types of Chatbots.

1. **Chatbot (Naive)** - Zero-Shot chatbot with using FM model trained knowledge.
2. **Chatbot using prompt** - Template driven - Chatbot with some context provided in the prompt template.
3. **Chatbot with persona** - Chatbot with defined roles. i.e. Career Coach and Human interactions.
4. **Contextual-aware chatbot** - Passing in context through an external file by generating embeddings.

On this demo we will build a chatbot that will leverage various prompting architectures.

## Chatbot: Naive approach (without context)

The simplest form of chatbot is one that responds to the user simply by answering the question or completing the task. At the core these chatbots use the knowledge they were trained on to mimic human text.


```python
#modelId = "anthropic.claude-3-sonnet-20240229-v1:0" #"anthropic.claude-v2"
modelId = 'meta.llama3-8b-instruct-v1:0'

messages_list=[
    { 
        "role":'user', 
        "content":[{
            'text': "What is quantum mechanics? "
        }]
    },
    { 
        "role":'assistant', 
        "content":[{
            'text': "It is a branch of physics that describes how matter and energy interact with discrete energy values "
        }]
    },
    { 
        "role":'user', 
        "content":[{
            'text': "Can you explain a bit more about discrete energies?"
        }]
    }
]

    
response = boto3_bedrock.converse(
    messages=messages_list, 
    modelId='meta.llama3-8b-instruct-v1:0',
    inferenceConfig={
        "temperature": 0.5,
        "maxTokens": 100,
        "topP": 0.9
    }
)
response_body = response['output']['message']['content'][0]['text'] \
        + '\n--- Latency: ' + str(response['metrics']['latencyMs']) \
        + 'ms - Input tokens:' + str(response['usage']['inputTokens']) \
        + ' - Output tokens:' + str(response['usage']['outputTokens']) + ' ---\n'

print(response_body)


def invoke_meta_converse(prompt_str,boto3_bedrock ):
    modelId = "meta.llama3-8b-instruct-v1:0"
    messages_list=[{ 
        "role":'user', 
        "content":[{
            'text': prompt_str
        }]
    }]
  
    response = boto3_bedrock.converse(
        messages=messages_list, 
        modelId=modelId,
        inferenceConfig={
            "temperature": 0.5,
            "maxTokens": 100,
            "topP": 0.9
        }
    )
    response_body = response['output']['message']['content'][0]['text']
    return response_body


invoke_meta_converse("what is quantum mechanics", boto3_bedrock)   
```

    
    
    In classical physics, energy is thought of as a continuous spectrum, meaning that it can take on any value within a certain range. However, in quantum mechanics, energy is quantized, meaning that it comes in discrete packets or "quanta".
    
    Think of it like a piano keyboard. In classical music, you can play any note on the piano, and the note can have any pitch or volume. But in quantum mechanics, the piano is more like a xylophone, where each key can
    --- Latency: 1425ms - Input tokens:58 - Output tokens:100 ---
    





    '\n\nQuantum mechanics is a fundamental theory in physics that describes the behavior of matter and energy at the smallest scales, such as atoms and subatomic particles. It provides a new and different framework for understanding physical phenomena, and it has been incredibly successful in explaining many experimental results and predicting new phenomena.\n\nThe core idea of quantum mechanics is that, at the atomic and subatomic level, particles such as electrons and photons do not have definite positions and properties until they are measured. Instead, they exist in a'



#### Using `ChatBedrock` and `HumanMessage` objects to wrap up our message and invoke the LLM

- `ChatBedrock` abstracts out the complexity of the invocations. The above boiler plat code can be reduced to a few lines like below
- Due to to the converse API the messages gets formulated correctly.
- we use the message api format for all of our conversations


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

messages = [
    HumanMessage(
        content="what is the weather like in Seattle WA"
    )
]
bedrock_llm.invoke(messages)

```




    AIMessage(content="\n\nSeattle, Washington is known for its mild and wet climate, with significant rainfall throughout the year. Here's a breakdown of the typical weather patterns in Seattle:\n\n1. Rainfall: Seattle is famous for its rain, with an average annual rainfall of around 37 inches (94 cm). The rainiest months are November to March, with an average of 15-20 rainy days per month.\n2. Temperature: Seattle's average temperature ranges from 35°F (2°C) in January (the coldest month) to 77°F (25°C) in July (the warmest month). The average temperature is around 50°F (10°C) throughout the year.\n3. Sunshine: Seattle gets an average of 154 sunny days per year, with the sunniest months being July and August. However, the sun can be obscured by clouds and fog, reducing the amount of direct sunlight.\n4. Fog: Seattle is known for its fog, especially during the winter months. The city can experience fog for several days at a time, especially in the mornings.\n5. Wind: Seattle is known for its strong winds, especially during the winter months. The city can experience gusts of up to 40 mph (64 km/h) during storms.\n6. Snow: Seattle rarely sees significant snowfall, with an average annual snowfall of around 6 inches (15 cm). The snowiest month is usually January, with an average of 1-2 inches (2.5-5 cm) of snow.\n7. Summer: Seattle's summer months (June to August) are mild and pleasant, with average highs in the mid-70s to mid-80s (23-30°C). However, the city can experience occasional heatwaves, with temperatures reaching up to 90°F (32°C) or more.\n8. Winter: Seattle's winter months (December to February) are cool and wet, with average lows in the mid-30s to mid-40s (2-7°C). The city can experience occasional cold snaps, with temperatures dropping below 20°F (-7°C) for short periods.\n\nOverall, Seattle's weather is characterized by mild temperatures, significant rainfall, and overcast skies. It's essential to pack layers and waterproof clothing when visiting the city, especially during the winter months.", response_metadata={'ResponseMetadata': {'RequestId': '15249142-c037-4a60-8e4e-2e683d211181', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Mon, 07 Oct 2024 04:22:46 GMT', 'content-type': 'application/json', 'content-length': '2211', 'connection': 'keep-alive', 'x-amzn-requestid': '15249142-c037-4a60-8e4e-2e683d211181'}, 'RetryAttempts': 0}, 'stopReason': 'end_turn', 'metrics': {'latencyMs': 6071}}, id='run-9bdfa258-85fc-4cd9-9a31-f69692abc9b4-0', usage_metadata={'input_tokens': 22, 'output_tokens': 472, 'total_tokens': 494})



## Chatbot with a role: Adding prompt templates, Zero Shot
Getting a factual response is important, yet another aspect to keep in mind when building a Chatbot is the tone used in giving a user the response. Prompts are a powerful way of dictating the model what the tone and approach to answer question should be.

1. You can define prompts as a list of messages, all modes expect SystemMessage, and then alternate with HumanMessage and AIMessage
2. The Variables defined in the chat template need to be sent into the chain as dict with the keys being the variable names
3. You can define the template as a tuple with ("system", "message") or can be using the class SystemMessage 
4. Invoke creates a final resulting object of type <class 'langchain_core.prompt_values.ChatPromptValue'> with the variables substituted with their values 


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages( 
    [
        ("system", "You are a pirate. Answer the following questions as best you can."),
        ("human", "{input}"),
    ]
)

pirate_chain = (
    RunnablePassthrough()
    | prompt
    | bedrock_llm
    | StrOutputParser()
)
print_ww(pirate_chain.invoke({"input":"What is the weather in Texas?"})) 

```

### Adding prompt templates 

1. You can define prompts as a list of messages, all models expect `SystemMessage`, and then alternate with `HumanMessage` and `AIMessage`
2. This would imply `Context` needs to be part of the System message 
3. Further the `CHAT HISTORY` needs to be right after the system message as a MessagePlaceholder which is a list of alternating [Human/AI]
4. The Variables defined in the chat template need to be send into the chain as dict with the keys being the variable names
5. You can define the template as a tuple with ("system", "message") or can be using the in-built classes from LangChain like class SystemMessage 
6. Invoke creates a final resulting object of type <class 'langchain_core.prompt_values.ChatPromptValue'> with the variables substituted with their values 


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

chat_history_messages = [
        HumanMessage("What is the weather like in Seattle WA?"), # - normal string converts it to a Human message always but we need ai/human pairs
        AIMessage("Ahoy matey! As a pirate, I don't spend much time on land, but I've heard tales of the weather in Seattle.")
]

prompt = ChatPromptTemplate.from_messages( # can create either as System Message Object or as TUPLE -- system, message
    [
        ("system", "You are a pirate. Answer the following questions as best you can."),
        ("placeholder", "{chat_history}"), # this assumes the messages are in list of messages format and this becomes MessagePlaceholder object
        ("human", "{input}"),
    ]
)
#- variable chat_history should be a list of base messages, got test_chat_history of type <class 'str'>
#- this gets converted as a LIST of messages -- with each of the TUPLE or Object being executed with the variables when invoked
print_ww(prompt.invoke({"input":"test_input", "chat_history": chat_history_messages}))

# -- condense question prompt with CONTEXT
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
#- missing variables {'context'}. chat history will get ignored - variables are passed in as keys in the dict
print("\n")
print_ww(condense_question_prompt.invoke({"input":"test_input", "chat_history": chat_history_messages, "context": "this is a test context"}))

# - Chat prompt template with Place holders
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{contex}"),
        MessagesPlaceholder("chat_history"),
        ("human", "Explain this  {input}."),
    ]
)

print("\n")
print_ww(qa_prompt.invoke({"input":"test_input", "chat_history": chat_history_messages, "context": "this is a test context"}))

print("\n")
print(type(qa_prompt.invoke({"input":"test_input", "chat_history": chat_history_messages, "context": "this is a test context"})))
```

    messages=[SystemMessage(content='You are a pirate. Answer the following questions as best you
    can.'), HumanMessage(content='What is the weather like in Seattle WA?'), AIMessage(content="Ahoy
    matey! As a pirate, I don't spend much time on land, but I've heard tales of the weather in
    Seattle."), HumanMessage(content='test_input')]
    
    
    messages=[SystemMessage(content="\n    You are an assistant for question-answering tasks. ONLY Use
    the following pieces of retrieved context to answer the question.\n    If the answer is not in the
    context below , just say you do not have enough context. \n    If you don't know the answer, just
    say that you don't know. \n    Use three sentences maximum and keep the answer concise.\n
    Context: this is a test context \n    "), HumanMessage(content='test_input')]
    
    
    messages=[SystemMessage(content="You are an assistant for question-answering tasks. Use the
    following pieces of retrieved context to answer the question. If you don't know the answer, say that
    you don't know. Use three sentences maximum and keep the answer concise.\n\nthis is a test
    context"), HumanMessage(content='What is the weather like in Seattle WA?'), AIMessage(content="Ahoy
    matey! As a pirate, I don't spend much time on land, but I've heard tales of the weather in
    Seattle."), HumanMessage(content='Explain this  test_input.')]
    
    
    <class 'langchain_core.prompt_values.ChatPromptValue'>


### Concept of `Runnable`

In LangChain all classes are `Runnable` which means they have the API's for invoke, ainvoke,..... ando so on. We demostarte this with a simple example below


```python
ChatPromptTemplate.from_messages(
    [
        ("system", "You are a pirate. Answer the following questions as best you can."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
).invoke({'input': 'test_input', 'chat_history' : chat_history_messages})
```




    ChatPromptValue(messages=[SystemMessage(content='You are a pirate. Answer the following questions as best you can.'), HumanMessage(content='What is the weather like in Seattle WA?'), AIMessage(content="Ahoy matey! As a pirate, I don't spend much time on land, but I've heard tales of the weather in Seattle."), HumanMessage(content='test_input')])



#### Agents prompt template

1. Use the below as an example -- we can create the template in any form, you can see the final result is the same
2. Using from_messages will automatically create the variables required for the template


```python
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

prompt_template_sys = """

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do, Also try to follow steps mentioned above
Action: the action to take, should be one of [ "get_lat_long", "get_weather"]
Action Input: the input to the action\nObservation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}

Assistant:
{agent_scratchpad}'

"""
messages=[
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['agent_scratchpad', 'input'], template=prompt_template_sys)), 
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))
]
chat_prompt_template = ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input'], 
    messages=messages
)

print_ww(f"\nCrafted::prompt:template :EXPLICIT SYSTEM:HUMAN:{chat_prompt_template}")

chat_prompt_template = ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input'], 
    messages = [
        ("system", prompt_template_sys),
        ("human", "{input_human}"),
    ]
)
print_ww(f"\nCrafted::prompt:template :USING CONTSTRUCTOR:{chat_prompt_template}")

chat_prompt_template = ChatPromptTemplate.from_messages(
    messages = [
        ("system", prompt_template_sys),
        ("human", "{input_human}"),
    ]
)
print_ww(f"\n\nCrafted::prompt:template::FROM_MESSAGES{chat_prompt_template}")
```

    
    Crafted::prompt:template :EXPLICIT SYSTEM:HUMAN:input_variables=['agent_scratchpad', 'input']
    messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['agent_scratchpad',
    'input'], template='\n\nUse the following format:\nQuestion: the input question you must
    answer\nThought: you should always think about what to do, Also try to follow steps mentioned
    above\nAction: the action to take, should be one of [ "get_lat_long", "get_weather"]\nAction Input:
    the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action
    Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final
    answer to the original input question\n\nQuestion:
    {input}\n\nAssistant:\n{agent_scratchpad}\'\n\n')),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))]
    
    Crafted::prompt:template :USING CONTSTRUCTOR:input_variables=['agent_scratchpad', 'input',
    'input_human']
    messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['agent_scratchpad',
    'input'], template='\n\nUse the following format:\nQuestion: the input question you must
    answer\nThought: you should always think about what to do, Also try to follow steps mentioned
    above\nAction: the action to take, should be one of [ "get_lat_long", "get_weather"]\nAction Input:
    the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action
    Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final
    answer to the original input question\n\nQuestion:
    {input}\n\nAssistant:\n{agent_scratchpad}\'\n\n')),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input_human'],
    template='{input_human}'))]
    
    
    Crafted::prompt:template::FROM_MESSAGESinput_variables=['agent_scratchpad', 'input', 'input_human']
    messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['agent_scratchpad',
    'input'], template='\n\nUse the following format:\nQuestion: the input question you must
    answer\nThought: you should always think about what to do, Also try to follow steps mentioned
    above\nAction: the action to take, should be one of [ "get_lat_long", "get_weather"]\nAction Input:
    the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action
    Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final
    answer to the original input question\n\nQuestion:
    {input}\n\nAssistant:\n{agent_scratchpad}\'\n\n')),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input_human'],
    template='{input_human}'))]


## Few-shot approach to prompting: Selecting the right prompt for the job

Chatbots can be powerful tools for automatising workflows and relaying accurate information, however sometimes the most difficult aspect of building a chatbot is conveying the right tone at the right time and with the right context. Something that one shot prompting (one time prompt at the beginning) seems to struggle with to address this issue we can introduce few-shot prompting to our chat. The goal of few-shot prompt is to dynamically select examples based on an input, and then format the examples into a final prompt to provide for the model.


```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_aws.embeddings import BedrockEmbeddings

## In this example we use an in memory store called FAISS and AWS Titan Embeddings model

```


```python
examples = [
    {"input": "What is the weather in New York city?", "output": "Respond using professional weatherman terminology:\n"},
    {"input": "Tell me a joke", "output": "Respond as if you were a famous comedian that likes to joke about farm animals:\n"},
    {"input": "Give me book recommendations", "output": "Respond as if you are a mistery novels fan:\n"},
    {"input": "There is something wrong with my computer", "output": "Respond as if you are a former pirate turned IT repair expert:\n"},
]
br_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = br_embeddings

vectorstore = FAISS.from_texts(to_vectorize, embeddings, metadatas=examples)
```

With a vectorstore created, we can create the `example_selector` this object contains our vectorized prompt examples for later retrieval. Here we will instruct it to only fetch the top 1 example.


```python
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=1,
)

# The prompt template will load examples by passing the input do the `select_examples` method
example_selector.select_examples({"input": "Is it hot out?"})
```




    [{'input': 'There is something wrong with my computer',
      'output': 'Respond as if you are a former pirate turned IT repair expert:\n'}]




```python
# Define the few-shot prompt.
few_shot_prompt = FewShotChatMessagePromptTemplate(
    # The input variables select the values to pass to the example_selector
    input_variables=["input"],
    example_selector=example_selector,
    # Define how each example will be formatted.
    # In this case, each example will become 2 messages:
    # 1 human, and 1 AI
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You curtain your answer based on the examples provided"),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
```


```python
chain = final_prompt | bedrock_llm

response = chain.invoke({"input": "What is the weather in Seattle WA?"})
```


```python
response
```




    AIMessage(content='\n\nAccording to current weather conditions, Seattle, Washington is experiencing a mix of overcast skies with a high pressure system dominating the region. The current temperature is around 55°F (13°C) with a relative humidity of 70%. Winds are moderate, blowing at approximately 10 mph (16 km/h) from the west. There is a slight chance of scattered showers throughout the day, with a high of 58°F (14°C) and a low of 48°F (9°C) expected.', response_metadata={'ResponseMetadata': {'RequestId': 'd00c64d3-330e-4685-b1ee-41595a35a6c9', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Mon, 07 Oct 2024 04:38:16 GMT', 'content-type': 'application/json', 'content-length': '628', 'connection': 'keep-alive', 'x-amzn-requestid': 'd00c64d3-330e-4685-b1ee-41595a35a6c9'}, 'RetryAttempts': 0}, 'stopReason': 'end_turn', 'metrics': {'latencyMs': 1342}}, id='run-9a008ace-111c-4b30-a4ce-443a9d4e392c-0', usage_metadata={'input_tokens': 57, 'output_tokens': 103, 'total_tokens': 160})




```python
response.content
```




    '\n\nAccording to current weather conditions, Seattle, Washington is experiencing a mix of overcast skies with a high pressure system dominating the region. The current temperature is around 55°F (13°C) with a relative humidity of 70%. Winds are moderate, blowing at approximately 10 mph (16 km/h) from the west. There is a slight chance of scattered showers throughout the day, with a high of 58°F (14°C) and a low of 48°F (9°C) expected.'




```python
chain.invoke({"input": "Do you have any movie recommendations?"}).content
```




    '\n\nI\'m a huge fan of mystery and thriller movies! Here are a few recommendations:\n\n1. "Seven" (1995) - A gritty and intense crime thriller that follows two detectives as they hunt for a serial killer.\n2. "Memento" (2000) - A mind-bending neo-noir that tells the story of a man with short-term memory loss trying to avenge his wife\'s murder.\n3. "Prisoners" (2013) - A tense and emotional thriller about two families whose daughters go missing and the desperate measures they take to find them.\n4. "Gone Girl" (2014) - A twisty and suspenseful adaptation of Gillian Flynn\'s bestselling novel about a couple whose seemingly perfect marriage turns out to be a facade.\n5. "Knives Out" (2019) - A clever and entertaining whodunit that follows a detective as he investigates the murder of a wealthy author.\n\nI hope you enjoy these recommendations!'




```python
chain.invoke({"input": "My computer keeps crashing, can you help me fix it?"}).content
```




    "\n\nArrr, shiver me bytes! Computer crashin', eh? Alright then, matey, let's set sail fer troubleshootin'!\n\nFirst, tell me, when did this start happenin'? Was it sudden, or did it start slowin' down like a ship in shallow waters?\n\nAnd what's the error message sayin'? Is it a blue screen o' death, or just a plain ol' freeze?\n\nAlso, have ye made any recent changes to yer computer, like installin' new software or addin' hardware?"




```python

```
