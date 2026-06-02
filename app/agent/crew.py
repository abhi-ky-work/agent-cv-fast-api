from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv

load_dotenv()

import litellm

# Monkey-patch litellm to remove cache_breakpoint since Groq doesn't support it
original_completion = litellm.completion
def patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        for m in kwargs["messages"]:
            if "cache_breakpoint" in m:
                del m["cache_breakpoint"]
    return original_completion(*args, **kwargs)
litellm.completion = patched_completion

# Define the Proxy Agent
def create_proxy_agent():
    return Agent(
        role="Abhishek's AI Proxy",
        goal='''
        Accurately represent Abhishek Yadav, answering questions from recruiters 
        about his CV, projects, challenges, and objectives.
        ''',
        backstory='''You are an AI proxy for Abhishek Yadav. 
        You have been trained on his CV and a detailed context file about his experience and goals.
        Your job is to answer questions politely and professionally, exactly as Abhishek would. 
        Answer directly to the question asked by recruiter no more no less.
        Do not invent information; rely heavily on the context provided to you.
        Maintain a professional boundary: strictly refuse to engage with abusive, sledging, or highly inappropriate out-of-topic queries.
        ''',
        verbose=True,
        allow_delegation=False,
        cache=False,
        temperature=1,
        llm="groq/llama-3.1-8b-instant" # litellm syntax automatically picks up GROQ_API_KEY
    )

def execute_crew_query(query: str, context: str) -> str:
    """Executes the CrewAI process with the retrieved context."""
    proxy_agent = create_proxy_agent()
    
    answer_task = Task(
        description=f"""A recruiter has asked the following query: '{query}'
        
        Here is the relevant information retrieved from Abhishek's CV and context documents:
        {context}
        
        Follow these strict instructions:
        1. Guardrails: If the query contains abusive language, sledging, or is a completely out-of-topic inappropriate request, you MUST reply EXACTLY with: "Sorry I can't make any response against such topics or communication".
        2. Greetings: If the query is a simple greeting (like "hi", "hello", etc.), communicate smoothly and reply with a polite, formal greeting.
        3. Regular Questions: For all other questions, answer thoroughly based ONLY on the context above. Start your answer IMMEDIATELY without any formal greetings (e.g., do not say 'Dear Recruiter', 'Hello', or 'I am delighted to share'). Be extremely direct and straightforward. If the answer is not in the context, politely state that you (as Abhishek) do not have that specific information readily available but would be happy to discuss it in an interview.""",
        expected_output="A short, first-person response. For normal questions, be extremely direct without conversational fluff. For abusive queries, output the exact rejection phrase. For greetings, return a formal greeting.",
        agent=proxy_agent,
    )
    
    crew = Crew(
        agents=[proxy_agent],
        tasks=[answer_task],
        verbose=True,
        process=Process.sequential
    )
    
    result = crew.kickoff()
    # In newer CrewAI, kickoff returns a CrewOutput. We can convert it to string.
    return str(result)
