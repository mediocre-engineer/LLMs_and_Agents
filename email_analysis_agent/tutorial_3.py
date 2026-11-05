"""Tutorial 3: the Gmail assistant as a real agent instead of a workflow.

In tutorials 1 and 2 we fixed the path in code: check spam, then move it or draft a reply.
Here the model gets a goal and four Gmail tools, and decides by itself which tool to call,
looks at the result, and picks the next step, until the job is done.

There is deliberately no "send" tool: the agent can only create drafts, which you review.
Uses the same credentials.json, token.json and .env as tutorial 2.
"""
import threading

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import tutorial_2 as gmail_api  # connect_to_gmail, to_message, save_draft, move_to_spam
from tutorial_1 import llm

gmail = None  # the Gmail API client, set in __main__
gmail_lock = threading.Lock()  # the agent can call several tools at once, in parallel threads,
                               # but one Gmail client is not thread-safe: one call at a time

GOAL = """You are an email assistant working in the user's Gmail inbox. Handle every unread email.
Always read an email before deciding what to do with it; a subject line is not enough.
- Spam, scams or phishing: move it to the Spam folder.
- Real email that needs an answer: write a short, friendly reply and save it as a draft.
  Don't invent facts, and never accept, decline or promise anything for the user: leave
  those choices in [brackets], like "Thursday at 3pm [works / doesn't work] for me".
- Notifications, newsletters and automated messages: leave them alone.
You cannot send email; the user reviews every draft.
Before each tool call, say in one short sentence what you're doing and why.
When you're done, give a one-line summary of what you did with each email."""


def fetch(email_id: str) -> dict:
    return gmail.users().messages().get(userId="me", id=email_id, format="full").execute()


# ---- tools: the model reads each docstring to decide when to use the tool ----

@tool
def list_unread_emails() -> list:
    """Lists up to 5 unread emails in the inbox, newest first, with their id, sender and subject."""
    with gmail_lock:
        found = gmail.users().messages().list(userId="me", q="is:unread in:inbox", maxResults=5).execute()
        emails = []
        for item in found.get("messages", []):
            msg = gmail_api.to_message(fetch(item["id"]))
            emails.append({"id": item["id"], "from": msg["sender"], "subject": msg["subject"]})
        return emails


@tool
def read_email(email_id: str) -> dict:
    """Returns the sender, recipient, subject and full text of one email."""
    with gmail_lock:
        return gmail_api.to_message(fetch(email_id))


@tool
def move_to_spam(email_id: str) -> str:
    """Moves an email to the Spam folder. Use only for spam, scams or phishing."""
    with gmail_lock:
        gmail_api.move_to_spam(gmail, email_id)
        return "Moved to the Spam folder."


@tool
def create_draft_reply(email_id: str, reply_text: str) -> str:
    """Saves reply_text as a draft reply to the email, in the same conversation.
    The draft is NOT sent; the user reviews it first."""
    with gmail_lock:
        raw = fetch(email_id)
        if raw["threadId"] in gmail_api.threads_with_drafts(gmail):
            return "This conversation already has a draft, so nothing was added."
        gmail_api.save_draft(gmail, raw, reply_text)
        return "Draft saved."


tools = [list_unread_emails, read_email, move_to_spam, create_draft_reply]
llm_with_tools = llm.bind_tools(tools)  # the model now knows each tool's name, inputs and description


# ---- the agent loop: think -> call tools -> think -> ... -> finish ----

def think(state: MessagesState):
    """The model reads everything so far and either calls tools or gives its final answer."""
    return {"messages": [llm_with_tools.invoke([SystemMessage(GOAL)] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("think", think)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "think")
builder.add_conditional_edges("think", tools_condition)  # tool calls -> "tools", otherwise finish
builder.add_edge("tools", "think")                       # after acting, think again

agent = builder.compile()


def show(update: dict):
    """Prints each step so you can watch the agent decide."""
    for node, change in update.items():
        for message in change["messages"]:
            if node == "think":
                if message.content:
                    print(f"\nAgent: {message.content}")
                for call in message.tool_calls:
                    args = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in call["args"].items())
                    print(f"  -> {call['name']}({args})")
            else:
                print(f"  <- {str(message.content)[:160]}")


if __name__ == "__main__":
    gmail = gmail_api.connect_to_gmail()
    request = {"messages": [("user", "Please handle my unread emails.")]}
    for update in agent.stream(request, {"recursion_limit": 40}, stream_mode="updates"):
        show(update)
