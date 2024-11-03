from dotenv import load_dotenv
from time import time
from components.graph import app
from loguru import logger
load_dotenv()

config = {"configurable": {"thread_id": "2"}}

def main():
    logger.info("System is ready for user queries.")
    while True:
        query = input("Your question: ")
        if query.lower() == 'exit':
            logger.info("User terminated the session.")
            break
        inputs = {"question": query}
        for event in app.stream(inputs, config):
            for _,values in event.items():
                continue
        #dict_keys(['question', 'generation', 'documents'])
        logger.info(f"Final response: {values.get('generation')}")

if __name__ == "__main__":
    main()
    