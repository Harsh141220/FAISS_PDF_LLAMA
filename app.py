import os
import sys
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Replicate
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import openai
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer,AutoModelForCausalLM



# Load environment variables
env='gcp-starter'
replicate='r8_RN2g#########################33' #your replicate key
os.environ['REPLICATE_API_TOKEN'] = replicate

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase
    
pdf_reader = PdfReader('path to your pdf')
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()
knowledgeBase = process_text(text)

def resp(query,chat_history):        
        if query:
            QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
            Chat History:
            {chat_history}
            Follow Up Input: {query}
            Standalone question:""")
            docs = knowledgeBase.similarity_search(query)
            llm = Replicate(
            model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
            input={"temperature": 0.5, "max_length": 5000,"top_p":0.9} 
            )
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query,chat_history=chat_history)                
            return(response)
            
chat_history = []
while True:
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result=resp(query,chat_history)
    print('Answer: ' + result + '\n')
    chat_history.append((query, result))
