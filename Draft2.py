# Togather AI  #9d8f9ecbeb24ec738d5e05bcdafedd6c3d347e600e8de72ab94921442aba68ae
# Mongo DB hxunqdiKYxWSRfjL
# Mongo DB mongodb+srv://dhucomputer:hxunqdiKYxWSRfjL@cluster0.ejkkfaw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
# Smith 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
from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

print("TOGETHER_API_KEY:", os.getenv("TOGETHER_API_KEY"))
print("MONGODB_URI:", os.getenv("MONGODB_URI"))

from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

# TEST: Print the Together API Key to verify it's loaded
print("TOGETHER_API_KEY:", os.getenv("TOGETHER_API_KEY"))
print("MONGODB_URI:", os.getenv("MONGODB_URI"))



# Load .env file
load_dotenv()

from langchain_together import ChatTogether

# Initialize TogetherAI model
llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# MongoDB setup
mongo_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongo_uri)
db = client["support_ai_db"]
collection = db["documents"]

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection,
    index_name="vector_index",
    relevance_score_fn="cosine",
)

# LangSmith tracing (optional)
os.environ["LANGSMITH_TRACING"] = "true"

# Document loading
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Add documents to vector store
_ = vector_store.add_documents(documents=all_splits)

# LangGraph setup
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve relevant documents from the vector store based on a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    recent_tool_messages = [m for m in reversed(state["messages"]) if m.type == "tool"]
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        m for m in state["messages"]
        if m.type in ("human", "system") or (m.type == "ai" and not m.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}

from langgraph.graph import END

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

# Example query
while True:
    input_message = input("\nYour question (type 'exit' to quit): ")
    if input_message.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()