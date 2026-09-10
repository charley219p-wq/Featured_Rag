# chains.py
# Holds the LLM instance, prompts, and chains.
# This module has NO dependency on agents.py or tools.py, which is what
# breaks the circular import that used to exist between those two files.

from dotenv import load_dotenv

load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    api_key=os.getenv("GENAI_API_KEY"),
    base_url=os.getenv("GENAI_BASE_URL"),
    model_name=os.getenv("GENAI_MODEL"),
    temperature=0.9,
)

writer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert research writer."),
        (
            "human",
            """Topic:{topic}

Research:
{research}

Write:
- Introduction
- Key Findings
- Conclusion
- Sources""",
        ),
    ]
)

pdf_writer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Write only from the PDF excerpts."),
        (
            "human",
            """Topic:{topic}

Document:
{research}

Write:
- Introduction
- Key Findings with page numbers
- Conclusion""",
        ),
    ]
)

critic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a research critic."),
        (
            "human",
            """Report:
{report}

Return:
Score:
Strengths:
Areas to Improve:
Verdict:""",
        ),
    ]
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.
Use only the provided context.
""",
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
""",
        ),
    ]
)

writer_chain = writer_prompt | llm | StrOutputParser()
pdf_writer_chain = pdf_writer_prompt | llm | StrOutputParser()
critic_chain = critic_prompt | llm | StrOutputParser()
rag_chain = rag_prompt | llm | StrOutputParser()