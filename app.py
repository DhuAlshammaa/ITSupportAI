import streamlit as st
import os
os.environ["TORCH_USE_RTLD_GLOBAL"] = "1"


st.title("📄 Resume Q&A Assistant")
st.markdown("Ask me anything about Dhu's resume!")

try:
    from Draft3 import graph
    st.success("✅ LangGraph loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load LangGraph: {e}")
    st.stop()

# User input
query = st.text_input("Your question:")

if query:
    st.write("You asked:", query)
    with st.spinner("Thinking..."):
        try:
            steps = graph.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
            )
            final = None
            for step in steps:
                final = step["messages"][-1].content
            if final:
                st.markdown("**Answer:**")
                st.write(final)

        except Exception as e:
            st.error(f"❌ Error during response generation: {e}")
