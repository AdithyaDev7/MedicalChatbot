from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_pinecone import PineconeVectorStore

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone

os.environ['PINECONE_API_KEY'] = '18f4156e-5f6a-4fb8-a38a-fe6aedd0f3c9'
api_key = os.environ['PINECONE_API_KEY']
os.environ['PINECONE_API_ENV'] = 'us-east-1'
api_env = os.environ['PINECONE_API_ENV']

index_name="medical-chatbot"

#Loading the index
docsearch=Pinecone.from_existing_index(index_name, embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="D:\Mini Project\End-to-end-Medical-Chatbot-using-Llama2\model\llama-2-7b-chat.ggmlv3.q2_K.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')

greeting_messages = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "good night"]
greeting_response = "Hello! How can I assist you today?"

# def qa(query_dict):
#     # Simulate the qa function, replace with actual implementation
#     return {"result": "This is a simulated answer for the query: " + query_dict["query"]}


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    input_message = msg.lower().strip()
    if any(greeting in input_message for greeting in greeting_messages):
        print("Response: ", greeting_response)
        return greeting_response
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug= True)


