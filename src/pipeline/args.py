from dataclasses import dataclass
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@dataclass
class args:
  DOCPATH = "./data/sample.txt"
  CHUNKSIZE = 1000
  CHUNKOVERLAP = 0
  embeddings = OpenAIEmbeddings()
  text_splitter = CharacterTextSplitter(chunk_size= CHUNKSIZE, chunk_overlap=CHUNKOVERLAP)
  ModelName = "gpt-4-0613"
  connection_string = "mongodb://localhost:27017"
  session_id = "data-base"








  

