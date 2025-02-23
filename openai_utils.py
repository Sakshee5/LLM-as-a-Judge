import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

if 'api_key' not in st.session_state:
        st.session_state.api_key = None

# Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=st.session_state.api_key)

def get_response(user_prompt, model, json_format=True):
    if json_format:
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': user_prompt}],
            response_format={"type":"json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        return completion.choices[0].message.content