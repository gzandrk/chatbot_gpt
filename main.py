import os
import openai

import streamlit as st
from dataclasses import dataclass
from typing import Literal
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

openai.api_key = os.getenv("OPENAI_API_KEY")

# read pdf
loader = PyPDFLoader("data/Guidebook.pdf")
pages = loader.load_and_split()
      
# create embeddings
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_documents(pages, embeddings)

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

if 'chatbot_history' not in st.session_state:
    st.session_state.chatbot_history = []

st.title("KeJepang AI🤖")

chat_placeholder = st.container()
prompt_placeholder = st. form("chat-form")
credit_card_placeholder = st.empty()

with prompt_placeholder:
    st.markdown("**Ask me🤖!** - _press Enter to Submit_")
    cols = st.columns((6,1))
    cols[0].text_input(
        "Chat",
        placeholder="Ask me!",
        value="",
        label_visibility="collapsed",
        key="user_question"
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
    )
    user_question = st.session_state.user_question    

    if user_question:
        st.session_state.chatbot_history.append({'question': user_question, 'answer': ''})
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            st.session_state.chatbot_history[-1]['answer'] = response
            print(cb)
            st.write(response)
            st.write('---')

    for i, data in enumerate(st.session_state.chatbot_history, 1):
        st.markdown(f'***You*** : {data["question"]}')
        if data["answer"]:
            st.markdown(f'***AI*** : {data["answer"]}')
            st.write('---')
            


          
