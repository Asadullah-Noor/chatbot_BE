import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langgraph.graph import (
    StateGraph,
    START,
    END,
)
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from pathlib import Path
from dotenv import load_dotenv
# =========================================================
# 1. ENVIRONMENT
# =========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing from .env"
    )
# =========================================================
# 2. LLM
# =========================================================
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0,
)
# =========================================================
# 3. LANGGRAPH STATE
# =========================================================
class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]
# =========================================================
# 4. CHATBOT NODE
# =========================================================
def chatbot(state: ChatState):

    print("\n========== CHATBOT NODE ==========")

    response = llm.invoke(
        state["messages"]
    )

    return {
        "messages": [
            response
        ]
    }


# =========================================================
# 5. BUILD LANGGRAPH
# =========================================================

builder = StateGraph(
    ChatState
)

builder.add_node(
    "chatbot",
    chatbot
)

builder.add_edge(
    START,
    "chatbot"
)

builder.add_edge(
    "chatbot",
    END
)


# =========================================================
# 6. CHECKPOINTER
# =========================================================

checkpointer = InMemorySaver()


# =========================================================
# 7. COMPILE GRAPH
# =========================================================

graph = builder.compile(
    checkpointer=checkpointer
)


# =========================================================
# 8. FASTAPI
# =========================================================

app = FastAPI(
    title="LangGraph Chatbot",
    version="1.0.0"
)


# =========================================================
# 9. CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# 10. CONVERSATION METADATA
# =========================================================

"""
This dictionary stores sidebar information.

Example:

{
    "thread-id": {
        "title": "What is LangGraph?",
        "created_at": "...",
        "updated_at": "..."
    }
}

The actual messages are stored by LangGraph's checkpointer.
"""

conversations = {}


# =========================================================
# 11. REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):

    message: str

    thread_id: str


class NewChatResponse(BaseModel):
    thread_id: str


# =========================================================
# 12. HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "LangGraph Chatbot Backend is running"
    }


# =========================================================
# 13. CREATE NEW CHAT
# =========================================================

@app.post("/new-chat",response_model=NewChatResponse)
def new_chat():

    thread_id = str(
        uuid.uuid4()
    )
    conversations[thread_id] = {

        "thread_id": thread_id,

        "title": "New Chat",

    }

    return {
        "thread_id": thread_id
    }


# =========================================================
# 14. GET ALL CONVERSATIONS
# =========================================================

@app.get(
    "/conversations"
)
def get_conversations():

    return {
        "conversations": list(conversations.values())
    }


# =========================================================
# 15. CHAT
# =========================================================

@app.post(
    "/chat"
)
def chat(
    request: ChatRequest
):

    try:

        thread_id = request.thread_id

        message = request.message.strip()

        if not message:

            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
        # -------------------------------------------------
        # Create metadata if thread doesn't exist
        # -------------------------------------------------
        if thread_id not in conversations:
            conversations[thread_id] = {
                "thread_id": thread_id,
                "title": "New Chat",
            }
        # -------------------------------------------------
        # Get previous conversation
        # -------------------------------------------------
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        # -------------------------------------------------
        # Check whether this is first message
        # -------------------------------------------------
        try:
            current_state = graph.get_state(
                config
            )
            existing_messages = (
                current_state.values.get(
                    "messages",
                    []
                )
            )
        except Exception:
            existing_messages = []
        # -------------------------------------------------
        # Generate title from first message
        # -------------------------------------------------
        if len(existing_messages) == 0:
            title = message[:40]
            if len(message) > 40:
                title += "..."
            conversations[
                thread_id
            ]["title"] = title
        # -------------------------------------------------
        # Invoke LangGraph
        # -------------------------------------------------

        result = graph.invoke(

            {
                "messages": [
                    HumanMessage(
                        content=message
                    )
                ]
            },

            config=config
        )


        # -------------------------------------------------
        # Get AI response
        # -------------------------------------------------

        messages = result["messages"]
        ai_message = None
        for message_item in reversed(messages):

            if isinstance(
                message_item,
                AIMessage
            ):

                ai_message = message_item

                break


        if ai_message is None:
            raise Exception(
                "AI response was not generated"
            )
        return {
            "thread_id": thread_id,
            "response": ai_message.content,
            "title":
                conversations[
                    thread_id
                ]["title"]
        }
    except HTTPException:

        raise


    except Exception as error:

        print(
            "CHAT ERROR:",
            error
        )

        raise HTTPException(

            status_code=500,

            detail=str(error)

        )


# =========================================================
# 16. GET ONE CONVERSATION HISTORY
# =========================================================

@app.get(
    "/history/{thread_id}"
)
def get_history(
    thread_id: str
):

    try:

        config = {

            "configurable": {

                "thread_id": thread_id

            }

        }


        state = graph.get_state(
            config
        )


        messages = []


        for message in state.values.get(
            "messages",
            []
        ):

            if isinstance(
                message,
                HumanMessage
            ):

                messages.append({

                    "role": "user",

                    "content":
                        message.content

                })


            elif isinstance(
                message,
                AIMessage
            ):

                messages.append({

                    "role": "assistant",

                    "content":
                        message.content

                })


        return {

            "thread_id": thread_id,

            "title":
                conversations.get(
                    thread_id,
                    {}
                ).get(
                    "title",
                    "Chat"
                ),

            "messages": messages

        }


    except Exception as error:

        print(
            "HISTORY ERROR:",
            error
        )

        raise HTTPException(

            status_code=500,

            detail=str(error)

        )


# =========================================================
# 17. GET CURRENT LANGGRAPH STATE
# =========================================================

@app.get(
    "/state/{thread_id}"
)
def get_state(
    thread_id: str
):

    try:

        config = {

            "configurable": {

                "thread_id": thread_id

            }

        }


        state = graph.get_state(
            config
        )


        return {

            "thread_id": thread_id,

            "messages": [

                message.content

                for message

                in state.values.get(
                    "messages",
                    []
                )

            ],

            "next": state.next,

            "checkpoint_id":
                state.config[
                    "configurable"
                ].get(
                    "checkpoint_id"
                )

        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)

        )


# =========================================================
# 18. CHECKPOINT HISTORY
# =========================================================

@app.get(
    "/checkpoints/{thread_id}"
)
def get_checkpoints(
    thread_id: str
):

    try:

        config = {

            "configurable": {

                "thread_id": thread_id

            }

        }


        history = list(

            graph.get_state_history(
                config
            )

        )


        checkpoints = []


        for state in history:

            checkpoints.append({

                "checkpoint_id":
                    state.config[
                        "configurable"
                    ].get(
                        "checkpoint_id"
                    ),

                "values": {

                    "messages": [

                        message.content

                        for message

                        in state.values.get(
                            "messages",
                            []
                        )

                    ]

                },

                "next": state.next

            })


        return {

            "thread_id": thread_id,

            "checkpoints":
                checkpoints

        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)

        )