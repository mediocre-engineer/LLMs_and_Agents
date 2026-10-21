"""Tutorial 2: run the email agent on real, unread Gmail messages.

Spam is moved to the Gmail Spam folder. For real mail, the agent from tutorial 1 gets one
more node, write_reply, and each reply is saved as a Gmail draft in the same thread, so you
can review, edit and send it yourself. Nothing is ever sent by this script.

Setup (once):
  1. In Google Cloud Console, enable the Gmail API and create an OAuth client ID
     of type "Desktop app". Download it as credentials.json into this folder.
  2. Run this script. A browser window asks you to allow access; the login is then
     saved to token.json so you only do this once.

Never commit credentials.json, token.json or .env: they give access to your accounts.
"""
import base64
import os
import re
from email.message import EmailMessage
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langgraph.graph import END, START, StateGraph

from tutorial_1 import (InboxState, alert_recipient, as_text, ask_llm, check_spam, choose_path,
                        discard_spam, load_message, summarize)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",   # read emails and move spam (no permanent delete)
    "https://www.googleapis.com/auth/gmail.compose",  # create drafts (this script never sends)
]
MAX_BODY_CHARS = 4000  # long emails are cut so each model call stays small


# ---- agent: tutorial 1's graph with write_reply between summarize and alert_recipient ----

class ReplyState(InboxState):
    reply: Optional[str]  # set by write_reply; None when no reply is needed


def write_reply(state: ReplyState):
    prompt = (
        "Write a short, friendly reply to the email below, written as the person who received it.\n"
        "Don't invent facts, dates or promises. Put anything the writer must fill in inside [brackets].\n"
        "If the email needs no reply (a notification, newsletter or automated message), answer only NO_REPLY.\n\n"
        + as_text(state["message"])
    )
    reply = ask_llm(prompt)
    needed = reply.strip().upper() != "NO_REPLY"
    print("Reply drafted." if needed else "No reply needed.")
    return {"reply": reply if needed else None, "history": state["history"] + [prompt, reply]}


builder = StateGraph(ReplyState)
builder.add_node("load_message", load_message)
builder.add_node("check_spam", check_spam)
builder.add_node("discard_spam", discard_spam)
builder.add_node("summarize", summarize)
builder.add_node("write_reply", write_reply)
builder.add_node("alert_recipient", alert_recipient)

builder.add_edge(START, "load_message")
builder.add_edge("load_message", "check_spam")
builder.add_conditional_edges(
    "check_spam",
    choose_path,
    {"spam": "discard_spam", "not_spam": "summarize"},
)
builder.add_edge("summarize", "write_reply")
builder.add_edge("write_reply", "alert_recipient")
builder.add_edge("discard_spam", END)
builder.add_edge("alert_recipient", END)

agent = builder.compile()


# ---- Gmail: log in, read unread emails, move spam, save drafts ----

def connect_to_gmail():
    """Logs in with OAuth and returns a Gmail API client. The login is saved to token.json."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json")
    if not creds or not creds.has_scopes(SCOPES):
        creds = None  # no login yet, or it was approved for fewer permissions: ask again
    elif not creds.valid and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
    with open("token.json", "w") as f:
        f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def find_text(part: dict) -> str:
    """Returns the plain-text body of a message, searching nested parts."""
    if part.get("mimeType") == "text/plain" and part["body"].get("data"):
        return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
    for child in part.get("parts", []):
        text = find_text(child)
        if text:
            return text
    return ""


def to_message(raw: dict) -> dict:
    """Converts a Gmail API message into the format the agent expects."""
    headers = {h["name"].lower(): h["value"] for h in raw["payload"]["headers"]}
    text = find_text(raw["payload"]) or raw.get("snippet", "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()[:MAX_BODY_CHARS]
    return {
        "sender": headers.get("from", "unknown"),
        "to": headers.get("to", "me"),
        "subject": headers.get("subject", "(no subject)"),
        "text": text,
    }


def save_draft(gmail, raw: dict, reply_text: str):
    """Saves reply_text as a draft answer to raw, in the same Gmail thread."""
    headers = {h["name"].lower(): h["value"] for h in raw["payload"]["headers"]}
    draft = EmailMessage()
    draft["To"] = headers.get("reply-to") or headers.get("from")
    subject = headers.get("subject", "")
    draft["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    if "message-id" in headers:  # these two headers keep the reply in the same conversation
        draft["In-Reply-To"] = headers["message-id"]
        draft["References"] = f"{headers.get('references', '')} {headers['message-id']}".strip()
    draft.set_content(reply_text)
    encoded = base64.urlsafe_b64encode(draft.as_bytes()).decode()
    body = {"message": {"raw": encoded, "threadId": raw["threadId"]}}
    gmail.users().drafts().create(userId="me", body=body).execute()


def move_to_spam(gmail, message_id: str):
    """Moves an email from the inbox to the Spam folder. It can be restored from there."""
    labels = {"addLabelIds": ["SPAM"], "removeLabelIds": ["INBOX"]}
    gmail.users().messages().modify(userId="me", id=message_id, body=labels).execute()


def threads_with_drafts(gmail) -> set:
    """Thread ids that already have a draft, so a second run doesn't draft twice."""
    drafts = gmail.users().drafts().list(userId="me").execute().get("drafts", [])
    return {d["message"]["threadId"] for d in drafts}


if __name__ == "__main__":
    gmail = connect_to_gmail()
    drafted = threads_with_drafts(gmail)
    found = gmail.users().messages().list(userId="me", q="is:unread in:inbox", maxResults=5).execute()
    print(f"Found {len(found.get('messages', []))} unread email(s).\n")
    for item in found.get("messages", []):
        if item["threadId"] in drafted:
            print("Skipping an email that already has a draft reply.")
            continue
        raw = gmail.users().messages().get(userId="me", id=item["id"], format="full").execute()
        result = agent.invoke({"message": to_message(raw), "spam": None, "verdict": None,
                               "summary": None, "reply": None, "history": []})
        if result["spam"]:
            move_to_spam(gmail, item["id"])
            print("Moved to the Spam folder.")
        elif result["reply"]:
            save_draft(gmail, raw, result["reply"])
            print(f"Draft saved in Gmail:\n{result['reply']}")
        print("-" * 40)
