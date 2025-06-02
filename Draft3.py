from dotenv import load_dotenv


import os
import re

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

from langchain_together import ChatTogether
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load TogetherAI model
llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

# Embeddings and MongoDB setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["support_ai_db"]
collection = db["documents"]
vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection,
    index_name="vector_index",
    relevance_score_fn="cosine",
)

# Load and clean resume
resume_path = "D:/projects/ITSupportAI/DhuAlshammaaResume.pdf"
resume_loader = PyMuPDFLoader(resume_path)
resume_docs = resume_loader.load()
for doc in resume_docs:
    doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
    doc.metadata["source"] = "DhuAlshammaaResume.pdf"

job_metadata_list = [
    {"job_title": "IT Support Specialist", "company": "Mega Corporate Operations", "start_date": "2025-02", "end_date": "Present", "role_order": 1},
    {"job_title": "Machine Learning Engineer - Internship", "company": "101GenAI", "start_date": "2024-05", "end_date": "Present", "role_order": 2},
    {"job_title": "Data and Network Analyst", "company": "AT&T", "start_date": "2022-10", "end_date": "2024-05", "role_order": 3},
    {"job_title": "Data Analyst – AI Team", "company": "US Radiology", "start_date": "2017-10", "end_date": "2020-07", "role_order": 4},
]

resume_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
resume_splits = resume_splitter.split_documents(resume_docs)

for i, chunk in enumerate(resume_splits):
    chunk.metadata.update({"source": "DhuAlshammaaResume.pdf", "chunk_id": i})
    for job in job_metadata_list:
        if job["job_title"] in chunk.page_content and job["company"] in chunk.page_content:
            chunk.metadata.update(job)
            break

vector_store.add_documents(resume_splits)

# Load and chunk blog
blog_loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
)
blog_docs = blog_loader.load()
blog_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
blog_splits = blog_splitter.split_documents(blog_docs)
vector_store.add_documents(blog_splits)

# LangGraph setup
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve relevant documents from the vector store based on a query."""
    retrieved_docs = vector_store.similarity_search(query, k=8)

    # ✅ If vector search fails, do manual fallback
    print("Valled Retrival")
    if not retrieved_docs or all(len(doc.page_content.strip()) == 0 for doc in retrieved_docs):
        print("[DEBUG] Vector search failed. Using manual fallback...")
        fallback_docs = [
            doc for doc in resume_splits
            if "818" in doc.page_content or "220-3481" in doc.page_content
        ]
        if fallback_docs:
            print("[DEBUG] Manual fallback found result.")
        retrieved_docs = fallback_docs

    print(f"\n[DEBUG] Retrieved {len(retrieved_docs)} documents for query: '{query}'")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n[Doc {i}] Source: {doc.metadata.get('source', 'N/A')}")
        print(doc.page_content[:300])
        print("=" * 60)

    # serialized = "\n\n".join(
    #     (f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}")
    #     for doc in retrieved_docs
    seen_contents = set()
    unique_docs = []
    for doc in retrieved_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}")
        for doc in unique_docs
    )
    return serialized, unique_docs

   # return serialized, retrieved_docs


def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    recent_tool_messages = [m for m in reversed(state["messages"]) if m.type == "tool"]
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(m.content for m in tool_messages)

    if not docs_content.strip():
        return {"messages": [{
            "type": "ai",
            "content": "Not found in resume."
        }]}

    system_message_content = (
        "You are a smart resume assistant. Use ONLY the retrieved resume content below to answer the user's question. "
        "Search the text for details like names, emails, phone numbers, job titles, employers, or dates. "
        "Give exact answers. If the answer is not in the retrieved content, say 'Not found in resume.' "
        "\n\nRetrieved Resume Content:\n"
        f"{docs_content}"
    )

    print("\n[DEBUG] SYSTEM MESSAGE TO LLM:\n")
    print(system_message_content)

    conversation_messages = [
        m for m in state["messages"]
        if m.type in ("human", "system") or (m.type == "ai" and not m.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [{
        "type": "ai",
        "content": response.content  # only show the answer, not the sources
    }]}



graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


for i, chunk in enumerate(resume_splits):
    if "818" in chunk.page_content or "220-3481" in chunk.page_content:
        print(f"\n✅ Phone number found in Chunk {i}")
        print(chunk.page_content)

# Interactive test
# while True:
#     input_message = input("\nYour question (type 'exit' to quit): ")
#     if input_message.lower() in {"exit", "quit"}:
#         print("Goodbye!")
#         break
#
#     for step in graph.stream(
#         {"messages": [{"role": "user", "content": input_message}]},
#         stream_mode="values",
#     ):
#         step["messages"][-1].pretty_print()

if __name__ == "__main__":
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

graph = graph_builder.compile()

resume_splits = resume_splitter.split_documents(resume_docs)
print(f"📄 Number of resume chunks: {len(resume_splits)}")

blog_splits = blog_splitter.split_documents(blog_docs)
print(f"📰 Number of blog chunks: {len(blog_splits)}")

for i, chunk in enumerate(resume_splits[:3]):  # Just first 3 chunks for preview
    print(f"\n--- Chunk {i} ---\n{chunk.page_content[:300]}")
