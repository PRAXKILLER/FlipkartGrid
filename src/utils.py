import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERP_API_KEY")
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import tiktoken
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import List
from langchain.prompts import StringPromptTemplate

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor,initialize_agent
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.document_loaders import TextLoader
from args import templates, args

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
import numpy as np
from langchain.memory import MongoDBChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler


# from utils import get_ruff_vectorstore, get_state_vectorstore, chat_gpt_agent, define_tools,define_agent
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

