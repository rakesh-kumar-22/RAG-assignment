import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from load_preprocess import docs, transactions
from retriever import retrieve_transactions
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(page_title="RAG Chatbot - Transactional Data", page_icon="🛒", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# Setup
@st.cache_resource
def setup_rag():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI()
    texts = [doc.page_content for doc in docs]
    embeddings = [embedding_model.embed_query(text) for text in texts]
    
    prompt_template = """
You are a retail transaction assistant.
Answer ONLY using the information in the provided context.

Rules:
1. If asked for purchase history, list each item with product name, amount (with ₹ symbol), and date.
2. If asked for total spending, sum all amounts and give answer like "[Customer] spent a total of ₹[amount]."
3. Do not assume or add any information not present in the context.
4. Use ₹ symbol for currency.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return embedding_model, llm, texts, embeddings, prompt

embedding_model, llm, texts, embeddings, prompt = setup_rag()

def answer_question(question):
    relevant_texts = retrieve_transactions(question, embeddings, texts, top_k=3)
    context = "\n".join(relevant_texts)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

# Sidebar - Analytics
st.sidebar.title("📊 Analytics Dashboard")

df = pd.DataFrame(transactions)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.strftime('%Y-%m')

# Chart: Spend per month
monthly_spend = df.groupby('month')['amount'].sum().reset_index()
fig1 = px.bar(monthly_spend, x='month', y='amount', 
              title='Spending per Month',
              labels={'amount': 'Amount (₹)', 'month': 'Month'},
              color='amount', color_continuous_scale='Blues')
st.sidebar.plotly_chart(fig1, use_container_width=True)

# Chart: Spend per customer
customer_spend = df.groupby('customer')['amount'].sum().reset_index()
fig2 = px.pie(customer_spend, values='amount', names='customer',
              title='Spending by Customer')
st.sidebar.plotly_chart(fig2, use_container_width=True)

# Main chat
st.title("🛒 RAG-Powered Chatbot for Transactional Data")
st.markdown("Ask questions about customer transactions!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state.last_question = user_input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer_question(user_input)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Memory feature
if st.session_state.last_question:
    st.sidebar.markdown("---")
    st.sidebar.subheader("💭 Memory")
    if st.sidebar.button("Show my last question"):
        st.sidebar.info(f"Last: {st.session_state.last_question}")

# Quick actions
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Quick Questions")
if st.sidebar.button("Riya's purchase history"):
    user_input = "Show me Riya's purchase history"
    st.session_state.last_question = user_input
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = answer_question(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

if st.sidebar.button("Amit's total spending"):
    user_input = "What is Amit's total spending?"
    st.session_state.last_question = user_input
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = answer_question(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
