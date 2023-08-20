import os
from dotenv import load_dotenv
load_dotenv()
import sys
# sys.path.append("./src/pipeline")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERP_API_KEY")
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import AgentType
import tiktoken
from langchain.tools import BaseTool
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import List
from langchain.prompts import StringPromptTemplate

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor,initialize_agent
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.document_loaders import TextLoader

from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import os
from langchain.memory import MongoDBChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler


# from utils import get_ruff_vectorstore, get_state_vectorstore, chat_gpt_agent, define_tools,define_agent
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import pandas as pd
from langchain.vectorstores import FAISS
# from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)

import os
 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis as RedisVectorStore
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from langchain.prompts.prompt import PromptTemplate
 
MAX_TEXT_LENGTH=1000  # Maximum num of text characters to use
 
def auto_truncate(val):
 
    """Truncate the given text."""
 
    return val[:MAX_TEXT_LENGTH]
 
# Load Product data and truncate long text fields
 
all_prods_df = pd.read_csv("D:\WORK\Flipkart Grid\ecommerce-assistant\ecommerce-assistant\data\product_data.csv", converters={
 
    'bullet_point': auto_truncate,
 
    'item_keywords': auto_truncate,
 
    'item_name': auto_truncate
 
})

all_prods_df['item_keywords'].replace('', None, inplace=True)
 
all_prods_df.dropna(subset=['item_keywords'], inplace=True)
 
# Reset pandas dataframe index
 
all_prods_df.reset_index(drop=True, inplace=True)


NUMBER_PRODUCTS = 2500  
 
# Get the first 2500 products
product_metadata = ( 
    all_prods_df
     .head(NUMBER_PRODUCTS)
     .to_dict(orient='index')
)
 
# Check one of the products
product_metadata[0]




 
# set your openAI api key as an environment variable
 
# data that will be embedded and converted to vectors
texts = [
    v['item_name'] for k, v in product_metadata.items()
]
 
# product metadata that we'll store along our vectors
metadatas = list(product_metadata.values())
 
# we will use OpenAI as our embeddings provider
embedding = OpenAIEmbeddings()
 
# name of the Redis search index to create
vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embedding,
)

print(vectorstore)


template = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
Or end the conversation if it seems like it's done.
Chat History:\"""
{chat_history}
\"""
Follow Up Input: \"""
{question}
\"""
Standalone question:"""
 
condense_question_prompt = PromptTemplate.from_template(template)
 
template = """You are a friendly, conversational retail shopping assistant. Use the following context including product names, descriptions, and keywords to show the shopper whats available, help find what they want, and answer any questions.
 
It's ok if you don't know the answer.
Context:\"""
 
{context}
\"""
Question:\"
\"""
 
Helpful Answer:"""
 
qa_prompt= PromptTemplate.from_template(template)


llm = OpenAI(temperature=0)
 
streaming_llm = OpenAI(
    streaming=True,
    verbose=True,
    max_tokens=150,
    temperature=0.2
)
 
# use the LLM Chain to create a question creation chain
question_generator = LLMChain(
    llm=llm,
    prompt=condense_question_prompt
)
 
# use the streaming LLM to create a question answering chain
doc_chain = load_qa_chain(
    llm=streaming_llm,
    chain_type="stuff",
    prompt=qa_prompt
)

chatbot = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    combine_docs_chain=doc_chain,
    question_generator=question_generator
)


# create a chat history buffer
# chat_history = []
# # gather user input for the first question to kick off the bot
# question = input("Hi! What are you looking for today?")
 
# # keep the bot running in a loop to simulate a conversation
# while True:
#     result = chatbot(
#         {"question": question, "chat_history": chat_history}
#     )
#     print("\n")
#     print(result)
#     chat_history.append((result["question"], result["answer"]))
#     question = input()



















