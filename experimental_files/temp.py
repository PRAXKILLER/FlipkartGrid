import sys
from typing import Dict
sys.path.append("./src")
sys.path.append("./src/pipeline")
sys.path.append("./app")


import requests


# payload = {
#   "session_id": "string",
#   "user_id": "string",
#   "chat_history": [

#   ],
#   "query": "What is Chatgpt",
#   "message_id": "string"
# }
# response = requests.post("http://0.0.0.0:7000/answer_query", json = payload)

# print(response.content)

class TempRun:
    def __init__(self):
        self.url = "http://0.0.0.0:7000/answer_query"
    
    def run(self, chat_history, question):
        payload = {
            "session_id": "string",
            "user_id": "string",
            "chat_history": chat_history,
            "query": question,
            "message_id": "string"
            }
        response = requests.post(self.url, json = payload)
        return response


fetch_answer = TempRun()
chat_history = []


while True:
    question = input("Enter question: ")
    response = fetch_answer.run(chat_history,question)
    # print(response.json()["answer"])
    # chat_history.extend(response.content["answer"])
    # print(response.json()["answer"])
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": response.json()["answer"]})
    print(chat_history)
    
