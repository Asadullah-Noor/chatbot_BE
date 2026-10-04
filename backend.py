import os

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


# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(
    "KEY PREFIX:",
    GROQ_API_KEY[:4]
)
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing from .env file"
    )

print(
    "API KEY LOADED:",
    bool(GROQ_API_KEY)
)
# =========================================================
# 2. LLM
# =========================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
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
# 5. BUILD GRAPH
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
# 8. FASTAPI APP
# =========================================================

app = FastAPI(
    title="LangGraph React Chatbot",
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

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# =========================================================
# 10. REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):

    message: str

    thread_id: str


# =========================================================
# 11. CHAT RESPONSE MODEL
# =========================================================

class ChatResponse(BaseModel):

    response: str

    thread_id: str


# =========================================================
# 12. HOME / HEALTH CHECK
# =========================================================

@app.get("/")
def home():

    return {
        "message": "LangGraph Chatbot Backend is running"
    }


# =========================================================
# 13. CHAT ENDPOINT
# =========================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    try:

        # -------------------------------------------------
        # Thread configuration
        # -------------------------------------------------

        config = {
            "configurable": {
                "thread_id": request.thread_id
            }
        }


        # -------------------------------------------------
        # Invoke LangGraph
        # -------------------------------------------------

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=request.message
                    )
                ]
            },

            config=config
        )


        # -------------------------------------------------
        # Get last AI message
        # -------------------------------------------------

        last_message = result[
            "messages"
        ][-1]


        # -------------------------------------------------
        # Return response
        # -------------------------------------------------

        return ChatResponse(

            response=last_message.content,

            thread_id=request.thread_id

        )


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
# 14. GET CHAT HISTORY
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


        # -------------------------------------------------
        # Get current state
        # -------------------------------------------------

        state = graph.get_state(
            config
        )


        messages = []


        # -------------------------------------------------
        # Convert LangChain messages
        # into frontend-friendly JSON
        # -------------------------------------------------

        for message in state.values.get(
            "messages",
            []
        ):

            if isinstance(
                message,
                HumanMessage
            ):

                role = "user"


            elif isinstance(
                message,
                AIMessage
            ):

                role = "assistant"


            else:

                continue


            messages.append({

                "role": role,

                "content": message.content

            })


        return {

            "thread_id": thread_id,

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
# 15. GET CURRENT GRAPH STATE
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

            "next": state.next,

            "checkpoint_id":
                state.config[
                    "configurable"
                ].get(
                    "checkpoint_id"
                )

        }


    except Exception as error:

        print(
            "STATE ERROR:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================================
# 16. CHECKPOINT HISTORY
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

            "checkpoints": checkpoints

        }


    except Exception as error:

        print(
            "CHECKPOINT ERROR:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )