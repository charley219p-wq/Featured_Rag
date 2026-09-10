# Merged pipeline.py
from langchain_core.messages import HumanMessage
from agents import build_pdf_agent, build_reader_agent,build_search_agent, writer_chain, critic_chain, pdf_writer_chain, rag_chain
from tools import load_pdf_to_vectorstore, get_full_pdf_summary_context, rag_search

def run_research_pipeline(topic):
    state={}
    search=build_search_agent()
    s=search.invoke({"messages":[HumanMessage(content=f"Find reliable information about {topic}")]})
    state["search_results"]=s["messages"][-1].content

    reader=build_reader_agent()
    r=reader.invoke({"messages":[HumanMessage(content=f"Pick the best URL and scrape it.\n{state['search_results']}")]})
    state["scraped_content"]=r["messages"][-1].content

    research=f"{state['search_results']}\n\n{state['scraped_content']}"
    state["report"]=writer_chain.invoke({"topic":topic,"research":research})
    state["feedback"]=critic_chain.invoke({"report":state["report"]})
    return state

def run_pdf_research_pipeline(pdf_path,topic=None):
    state={}
    load_pdf_to_vectorstore(pdf_path)
    query=topic or "Summarize this document"

    agent=build_pdf_agent()
    r=agent.invoke({"messages":[HumanMessage(content=f"Use pdf_search for {query}")]})
    state["retrieved_content"]=r["messages"][-1].content

    research=state["retrieved_content"]+"\n\n"+get_full_pdf_summary_context()
    state["report"]=pdf_writer_chain.invoke({"topic":query,"research":research})
    state["feedback"]=critic_chain.invoke({"report":state["report"]})
    return state

def ask_pdf_question(pdf_path, topic):
    state={}
    load_pdf_to_vectorstore(pdf_path)
    answer = rag_search(topic)
    return answer
if __name__=="__main__":
    while True:
        mode=input("1.Online 2.PDF 3.rag: ")
        if mode == "0":
            break
        if mode=="2":
            while True:
                pdf=input("PDF path: ")
                if pdf == "0":
                    break
                topic=input("Focus (Enter for summary): ").strip() or None
                out=run_pdf_research_pipeline(pdf,topic)
        elif mode == "3":
            pdf_path = input("PDF Path: ")

            load_pdf_to_vectorstore(pdf_path)

            while True:
                question = input("Question: ")

                if question.lower() == "exit":
                    break

                answer = rag_chain.invoke({
                                "question": question,
                                "context": context
                            })
                        
                
                
        else:
            topic=input("Topic: ")
            out=run_research_pipeline(topic)
        print(out["report"])
        print("\\n--- FEEDBACK ---\\n")
        print(out["feedback"])
        print(out["Answer"])

       