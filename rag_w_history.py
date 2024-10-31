import bs4
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define the SQLite database
DATABASE_URL = "sqlite:///chat_history.db"
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")

# Create the database and the tables
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to save a single message
def save_message(session_id: str, role: str, content: str):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()

# Function to load chat history
def load_session_history(session_id: str) -> BaseChatMessageHistory:
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            for message in session.messages:
                chat_history.add_message({"role": message.role, "content": message.content})
    except SQLAlchemyError:
        pass
    finally:
        db.close()

    return chat_history

# Modify the get_session_history function to use the database
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = load_session_history(session_id)
    return store[session_id]

# Ensure you save the chat history to the database when needed
def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])

# Example of saving all sessions before exiting the application
import atexit
atexit.register(save_all_sessions)

### Construct retriever ###
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
store = {}

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Invoke the chain and save the messages after invocation
def invoke_and_save(session_id, input_text):
    # Save the user question with role "human"
    save_message(session_id, "human", input_text)
    
    result = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    # Save the AI answer with role "ai"
    save_message(session_id, "ai", result)
    return result