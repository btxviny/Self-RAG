import os
from langchain_openai import AzureChatOpenAI

import dotenv
dotenv.load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.environ['API_BASE'],
    openai_api_version=os.environ['API_VERSION'],
    openai_api_key=os.environ['API_KEY'],
    deployment_name=os.environ['GPT_DEPLOYMENT_NAME'],
    openai_api_type="azure"
)