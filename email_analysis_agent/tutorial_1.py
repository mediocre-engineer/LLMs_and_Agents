"""Email spam agent: checks each email for spam and summarizes the legitimate ones."""
import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


class InboxState(TypedDict):
    message: dict           # sender, to, subject, text
    spam: Optional[bool]    # set by check_spam
    verdict: Optional[str]  # the model's spam verdict and reason
    summary: Optional[str]  # set by summarize, for real mail only
    history: list           # every prompt and reply, in order


def ask_llm(prompt: str) -> str:
    """Sends one prompt to the model and returns the reply text."""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def as_text(msg: dict) -> str:
    return f"From: {msg['sender']}\nSubject: {msg['subject']}\n\n{msg['text']}"


# ---- nodes: each takes the state and returns only the keys it changes ----

def load_message(state: InboxState):
    msg = state["message"]
    print(f"New email for {msg['to']}: '{msg['subject']}' from {msg['sender']}")
    return {}


def check_spam(state: InboxState):
    prompt = (
        "Decide whether the email below is spam.\n"
        "Reply with one word first, SPAM or NOT_SPAM, then one sentence explaining why.\n\n"
        + as_text(state["message"])
    )
    reply = ask_llm(prompt)
    print(f"Verdict: {reply}")
    first_word = reply.split()[0].strip("*.,:").upper()
    return {
        "spam": first_word == "SPAM",
        "verdict": reply,
        "history": state["history"] + [prompt, reply],
    }


def discard_spam(state: InboxState):
    print("Marked as spam. No summary needed.")
    return {}


def summarize(state: InboxState):
    prompt = (
        "Summarize the email below in two sentences or fewer. "
        "Focus on what the reader needs to know or do.\n\n" + as_text(state["message"])
    )
    summary = ask_llm(prompt)
    return {"summary": summary, "history": state["history"] + [prompt, summary]}


def alert_recipient(state: InboxState):
    msg = state["message"]
    print(f"Hi {msg['to']}, you have a new email from {msg['sender']}.")
    print(f"Subject: {msg['subject']}")
    print(f"Summary: {state['summary']}")
    return {}


def choose_path(state: InboxState) -> str:
    return "spam" if state["spam"] else "not_spam"


# ---- graph: wire the nodes together ----

builder = StateGraph(InboxState)
builder.add_node("load_message", load_message)
builder.add_node("check_spam", check_spam)
builder.add_node("discard_spam", discard_spam)
builder.add_node("summarize", summarize)
builder.add_node("alert_recipient", alert_recipient)

builder.add_edge(START, "load_message")
builder.add_edge("load_message", "check_spam")
builder.add_conditional_edges(
    "check_spam",
    choose_path,
    {"spam": "discard_spam", "not_spam": "summarize"},
)
builder.add_edge("summarize", "alert_recipient")
builder.add_edge("discard_spam", END)
builder.add_edge("alert_recipient", END)

agent = builder.compile()


if __name__ == "__main__":
    samples = [
        {
            "sender": "prize-desk@lucky-draw.win",
            "to": "Sam",
            "subject": "Claim your free phone today",
            "text": "Congratulations! You were selected. Pay a $4.99 shipping fee within 24 hours to claim your prize.",
        },
        {
            "sender": "Priya (Northwind Labs)",
            "to": "Sam",
            "subject": "Dataset for Friday's demo",
            "text": "Hi Sam, I've uploaded the cleaned dataset to the shared drive. "
                    "Could you run the evaluation before Friday's demo and send me the numbers?",
        },
    ]
    for email in samples:
        agent.invoke({"message": email, "spam": None, "verdict": None, "summary": None, "history": []})
        print("-" * 40)
