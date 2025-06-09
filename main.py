from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
from vector import retriever


template = """
You are an expert in answering questions about the content of the pdf file.

Here are some relevant content: {reviews}

Here is the question to answer: {question}
Use the content to answer the question.
"""

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


if __name__ == "__main__":
  while True:
    query = input("Enter your query: ")
    if query.lower() == "exit":
      break
    content = retriever.invoke(query)
    result = chain.invoke({"reviews": content, "question": query})
    print(result.content)
