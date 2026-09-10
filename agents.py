# Merged agents.py
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

from tools import web_search,scrape_url,pdf_search

llm = ChatOpenAI(
api_key=os.getenv("GENAI_API_KEY"),
base_url=os.getenv("GENAI_BASE_URL"),
model_name=os.getenv("GENAI_MODEL"),
temperature=0.9
)

def build_search_agent():
    return create_agent(model=llm,tools=[web_search])

def build_reader_agent():
    return create_agent(model=llm,tools=[scrape_url])

def build_pdf_agent():
    return create_agent(model=llm,tools=[pdf_search])

writer_prompt=ChatPromptTemplate.from_messages([
("system","You are an expert research writer."),
("human","""Topic:{topic}

Research:
{research}

Write:
- Introduction
- Key Findings
- Conclusion
- Sources""")
])

pdf_writer_prompt=ChatPromptTemplate.from_messages([
("system","Write only from the PDF excerpts."),
("human","""Topic:{topic}

Document:
{research}

Write:
- Introduction
- Key Findings with page numbers
- Conclusion""")
])

critic_prompt=ChatPromptTemplate.from_messages([
("system","You are a research critic."),
("human","""Report:
{report}

Return:
Score:
Strengths:
Areas to Improve:
Verdict:""")
])

writer_chain=writer_prompt|llm|StrOutputParser()
pdf_writer_chain=pdf_writer_prompt|llm|StrOutputParser()
critic_chain=critic_prompt|llm|StrOutputParser()