from dotenv import load_dotenv
from pprint import pprint
from components.graph import app

load_dotenv()

question1 = "Which produces better results, CAIN or DIFRINT?"
inputs = {"question": question1}

for output in app.stream(inputs, config={"configurable": {"thread_id": "2"}}):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])
