import os
from langchain_openai import ChatOpenAI

import dotenv
dotenv.load_dotenv()

llm = ChatOpenAI(
        api_key = os.environ['OPENAI_API_KEY'],
        model ='gpt-4o-mini'
    )