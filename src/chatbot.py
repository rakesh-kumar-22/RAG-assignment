"""
Complete demonstration of the RAG chatbot with custom retrieve_transactions function
"""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from load_preprocess import docs
from retriever import retrieve_transactions
from dotenv import load_dotenv

load_dotenv()

# Setup
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI()

# Extract texts and create embeddings
texts = [doc.page_content for doc in docs]
embeddings = [embedding_model.embed_query(text) for text in texts]

# Prompt template
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

def answer_question(question):
    """Answer question using custom retrieve_transactions function"""
    # Retrieve relevant transactions
    relevant_texts = retrieve_transactions(question, embeddings, texts, top_k=3)
    context = "\n".join(relevant_texts)
    
    # Generate answer
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer

# Demo queries
print("=" * 60)
print("RAG Chatbot Demo - Using Custom retrieve_transactions Function")
print("=" * 60)

queries = [
    "Show me Riya's purchase history.",
    "What is Amit's total spending?",
    "List all transactions for January 2024.",
    "Which product was purchased most often?"
]

for query in queries:
    print(f"\nUser: {query}")
    answer = answer_question(query)
    print(f"Bot: {answer}")
    print("-" * 60)
