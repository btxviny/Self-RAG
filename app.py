from dotenv import load_dotenv
from time import time
from components.graph import app
from loguru import logger
load_dotenv()

if __name__ == "__main__":
    start = time()
    inputs = {"question": "What are some future directions for Deep Learning Video Stabilization?"}
    config = {"configurable": {"thread_id": "2"}}
    for event in app.stream(inputs, config):
        for node,values in event.items():
            continue
    logger.info(f"Time Elapsed {time() - start}.")
    #dict_keys(['question', 'generation', 'documents'])
    logger.info(f"Final response: {values.get('generation')}")
    