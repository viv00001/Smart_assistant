import os
from textwrap import dedent
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI  

# ---------- 1. LLM Setup ----------
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Add it to .env or os.environ.")

    return ChatOpenAI(
        model_name="gpt-3.5-turbo",  
        temperature=0.1,
        openai_api_key=api_key,
        max_tokens=4096
    )

# ---------- 2. Summarizer ----------
def quick_summary(chunks, llm=None):
    llm = llm or get_llm()

    summarizer = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        token_max=3000,
        verbose=False,
    )
    raw_summary = summarizer.run(chunks)

    words = raw_summary.split()
    return " ".join(words[:150]) + ("..." if len(words) > 150 else "")

# ---------- 3. QA Chain ----------
def answer_query(user_question, vector_store, llm=None, top_k=4):
    llm = llm or get_llm()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    relevant_docs = retriever.get_relevant_documents(user_question)

    if not relevant_docs:
        return "No relevant content found in the document."

    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    prompt = dedent(f"""
        You are a helpful assistant answering based strictly on the provided context.
        If the answer is not found, reply: "The answer is not available in the document."
        Include a quote from the document with a reference to where it was found.

        Context:
        {context}

        Question: {user_question}
    """).strip()

    return llm.predict(prompt)
