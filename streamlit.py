import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mymcq.utils import read_file,get_table_data
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_models import ChatOpenAI
from src.mymcq.mcqgen import generate_evaluate_chain
from src.mymcq.logger import logging

with open('','r') as file:
    RESPONSE_JSON = json.load(file)
    st.title("MCQs Creator Application with Langchain")
with st.form("user_inputs"):
    uploaded_file  = st.file_uploader("Upload Your PDF or TxT file here")
    mcq_count=st.number_input("No.of MCQs",min_value  = 3,max_value=54)

    subject=st.text_input("Insert Subject",max_chars=20)

    tone=st.text_input("Complexity level for question", max_chars=20, placeholder="simple")
    button=st.form_submit_button("Create MCQs")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Loading....."):
            try:
                text=read_file(uploaded_file)
                with get_openai_callback() as cb:
                    response=generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )
            except Exception as e:
                traceback.print_exception(type(0))
                st.error("Error")

            else: 
                
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response, dict):
                    quiz=response.get("quiz", None)
                    if quiz is not None:
                        table_data  = get_table_data(quiz)
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Erro in data")
                else:
                    st.write(response)