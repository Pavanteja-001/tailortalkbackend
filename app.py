import streamlit as st
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from datetime import datetime, timedelta
import pytz
import os
import httpx
import asyncio
import re
from dotenv import load_dotenv
from dateutil import parser

# --- LangGraph Agent Setup ---
class AgentState(TypedDict):
    messages: list
    intent: str
    suggested_slots: list
    confirmed_slot: str
    time_zone: str
    events: list

load_dotenv()

# Backend URL from environment variable
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")  # Default to local for testing

# Initialize Groq model
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# Define prompts (unchanged)
intention_prompt = ChatPromptTemplate.from_template("""
You are a conversational AI agent for booking appointments in India. Based on the user's message, identify their intent and return only the intent keyword. Possible intents are:
- book_appointment
- check_availability
- confirm_booking
- cancel
- unclear

User message: {message}
Current conversation: {history}

Respond with a single word: the identified intent.
""")

suggestion_prompt = ChatPromptTemplate.from_template("""
You are a conversational AI agent for booking appointments in India. Suggest up to 5 available time slots in Asia/Kolkata time zone, formatted as human-readable dates/times (e.g., "June 26, 2025, 2:00 PM"). If the user specifies specific times (e.g., "4 PM", "12 PM", "7 PM") or requests 'any slot', 'anytime', 'whole day', or 'all day', prioritize checking those times first and then provide all available slots for the day between 9 AM and 6 PM if no exact match is found. Avoid repeating previous suggestions.

User message: {message}
Available slots (Asia/Kolkata): {slots}
Current conversation: {history}

Respond with a natural language message suggesting up to 5 time slots or asking for clarification if needed.
""")

confirmation_prompt = ChatPromptTemplate.from_template("""
You are a conversational AI agent in India. Confirm the booking details in Asia/Kolkata time zone, formatted as human-readable (e.g., "June 26, 2025, 2:00 PM").

Example output:
"Your appointment is confirmed for {slot} (Asia/Kolkata). Thank you for booking!"

Selected slot: {slot}
User message: {selected_message}
Current conversation: {history}

Respond with a confirmation message including the booking details.
""")

cancel_prompt = ChatPromptTemplate.from_template("""
You are a scheduling assistant AI in India. List upcoming events in Asia/Kolkata time zone and ask the user to select one by number to cancel.

User message: {message}
Upcoming events: {events}
Current conversation: {history}

Example output:
"Please select an event to cancel:\n1. Meeting at June 26, 2025, 10:00 AM\n2. Appointment at June 27, 2025, 3:00 PM"

Respond with a message listing events with their indices or confirming cancellation.
""")

def identify_intent(state: AgentState) -> AgentState:
    """Identify the user's intent from their message."""
    message = state["messages"][-1]["content"]
    history = "\n".join(msg["content"] for msg in state["messages"][:-1])
    try:
        response = llm.invoke(intention_prompt.format(message=message, history=history))
        state["intent"] = response.content.strip()
    except Exception as e:
        state["messages"].append({"error": f"Error processing request: {str(e)}. Please try again.", "role": "assistant"})
        state["intent"] = "unclear"
    return state

def parse_user_date(message: str) -> tuple[datetime | None, datetime | None]:
    """Parse the user's message to extract a specific date or time range."""
    message = message.lower()
    now = datetime.now(pytz.UTC)
    today = now.date()
    
    day_map = {
        "today": today,
        "tomorrow": today + timedelta(days=1),
        "monday": today + timedelta(days=(0 - today.weekday()) % 7),
        "tuesday": today + timedelta(days=(1 - today.weekday()) % 7),
        "wednesday": today + timedelta(days=(2 - today.weekday()) % 7),
        "thursday": today + timedelta(days=(3 - today.weekday()) % 7),
        "friday": today + timedelta(days=(4 - today.weekday()) % 7),
        "saturday": today + timedelta(days=(5 - today.weekday()) % 7),
        "sunday": today + timedelta(days=(6 - today.weekday()) % 7)
    }
    
    start_date = None
    end_date = None
    for day, date in day_map.items():
        if day in message:
            start_date = datetime.combine(date, datetime.min.time(), tzinfo=pytz.UTC)
            end_date = start_date + timedelta(days=1)
            break
    
    date_pattern = r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),\s*(\d{4})"
    match = re.search(date_pattern, message)
    if match:
        month = match.group(1)
        day = int(match.group(2))
        year = int(match.group(3))
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
        }
        start_date = datetime(year, month_map[month.lower()], day, tzinfo=pytz.UTC)
        end_date = start_date + timedelta(days=1)
    
    if start_date and "afternoon" in message:
        start_date = start_date.replace(hour=12, minute=0, second=0, microsecond=0)
        end_date = start_date.replace(hour=18, minute=0, second=0, microsecond=0)
    elif "any slot" in message or "any time" in message or "whole day" in message or "all day" in message:
        start_date = start_date.replace(hour=9, minute=0, second=0, microsecond=0) if start_date else (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        end_date = start_date.replace(hour=18, minute=0, second=0, microsecond=0) if start_date else (now + timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)

    return start_date, end_date

async def check_availability_node(state: AgentState) -> AgentState:
    """Check availability of slots for a specific day or week."""
    time_zone = "Asia/Kolkata"
    tz = pytz.timezone(time_zone)
    message = state["messages"][-1]["content"].strip().lower()
    
    start_date, end_date = parse_user_date(message)
    if not start_date:
        start_date = (datetime.now(pytz.UTC) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        end_date = start_date + timedelta(days=7)
    
    async with httpx.AsyncClient() as client:
        try:
            start_str = start_date.astimezone(pytz.UTC).isoformat()
            end_str = end_date.astimezone(pytz.UTC).isoformat()
            response = await client.get(
                f"{BACKEND_URL}/availability?start={start_str}&end={end_str}"
            )
            response.raise_for_status()
            slots = response.json()["available_slots"]
            state["suggested_slots"] = [
                datetime.fromisoformat(slot.replace('Z', '+00:00')).astimezone(tz).isoformat()
                for slot in slots
            ]
            formatted_slots = [
                datetime.fromisoformat(slot.replace('Z', '+00:00')).astimezone(tz).strftime("%B %d, %Y, %I:%M %p")
                for slot in slots[:5]
            ]
            state["messages"].append({"content": f"Available slots in Asia/Kolkata: {', '.join(formatted_slots)}", "role": "assistant"})
            return state
        except httpx.HTTPStatusError as e:
            state["messages"].append({"error": f"Error checking availability: {str(e)}", "role": "assistant"})
            return state

async def suggest_slots(state: AgentState) -> AgentState:
    """Suggest available slots in Asia/Kolkata and handle greetings or time requests."""
    message = state["messages"][-1]["content"].strip().lower()
    history = "\n".join(msg["content"] for msg in state["messages"][:-1])
    slots = state["suggested_slots"]
    time_zone = "Asia/Kolkata"
    tz = pytz.timezone(time_zone)
    now = datetime.now(tz)

    # Handle basic greetings
    greetings = ["hi", "hello", "hey"]
    if any(greet in message for greet in greetings):
        state["messages"].append({"content": "Hi!", "role": "assistant"})
        return state
    if "good morning" in message and now.hour < 12:
        state["messages"].append({"content": "Good morning!", "role": "assistant"})
        return state

    if not slots:
        state["messages"].append({"content": "No available slots found. Please try a different time or date.", "role": "assistant"})
        return state

    # Handle specific time requests (e.g., 4pm, 12pm, 7pm)
    requested_hours = []
    time_match = re.finditer(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', message)
    for match in time_match:
        hour = int(match.group(1))
        meridian = match.group(3)
        if meridian == "pm" and hour != 12:
            hour += 12
        elif meridian == "am" and hour == 12:
            hour = 0
        requested_hours.append(hour)

    # Prioritize requested times
    prioritized_slots = []
    if requested_hours:
        for slot in slots:
            dt = datetime.fromisoformat(slot.replace('Z', '+00:00')).astimezone(tz)
            if dt.hour in requested_hours:
                prioritized_slots.append(slot)

    # If no exact matches or "anytime"/"whole day" requested, use all slots
    filtered_slots = prioritized_slots if prioritized_slots else slots
    if not prioritized_slots and ("anytime" in message or "whole day" in message or "all day" in message or "any slot" in message):
        filtered_slots = [slot for slot in slots if 9 <= datetime.fromisoformat(slot.replace('Z', '+00:00')).astimezone(tz).hour < 18]

    formatted_slots = [
        datetime.fromisoformat(slot.replace('Z', '+00:00')).astimezone(tz).strftime("%B %d, %Y, %I:%M %p")
        for slot in filtered_slots[:5]
    ]

    if not formatted_slots:
        state["messages"].append({"content": "No matching slots found. Would you like to see other time ranges?", "role": "assistant"})
        return state

    try:
        response = llm.invoke(suggestion_prompt.format(message=message, slots=formatted_slots, history=history))
        state["messages"].append({"content": response.content.strip(), "role": "assistant"})
    except Exception as e:
        state["messages"].append({"error": f"Error suggesting slots: {str(e)}. Please try again.", "role": "assistant"})
    return state

async def confirm_booking(state: AgentState) -> AgentState:
    """Confirm a booking in Asia/Kolkata."""
    message = state["messages"][-1]["content"].strip().lower()
    slots = state["suggested_slots"]
    time_zone = "Asia/Kolkata"
    tz = pytz.timezone(time_zone)
    selected_slot = None

    # Try match slot by index (e.g., "1", "slot 2", "first")
    for i, slot in enumerate(slots, 1):
        if str(i) in message or f"slot {i}" in message or ("first" in message and i == 1):
            selected_slot = slot
            break

    # Try match by specific time (e.g., "book at 2pm")
    if not selected_slot:
        try:
            requested_time = parser.parse(message, fuzzy=True).time()
            for slot in slots:
                dt = datetime.fromisoformat(slot.replace('Z', '+00:00')).astimezone(tz)
                if dt.hour == requested_time.hour and abs(dt.minute - requested_time.minute) <= 15:
                    selected_slot = slot
                    break
        except Exception:
            pass

    # Default fallback to first slot
    if not selected_slot and slots:
        selected_slot = slots[0]

    if not selected_slot:
        state["messages"].append({"content": "No available slots to confirm. Please check availability first.", "role": "assistant"})
        return state

    state["confirmed_slot"] = selected_slot

    try:
        formatted_slot = datetime.fromisoformat(selected_slot.replace('Z', '+00:00')).astimezone(tz).strftime("%B %d, %Y, %I:%M %p")
        response = llm.invoke(confirmation_prompt.format(
            slot=formatted_slot,
            selected_message=message,
            history="\n".join(msg["content"] for msg in state["messages"][:-1])
        ))
        state["messages"].append({"content": response.content.strip(), "role": "assistant"})
    except Exception as e:
        state["messages"].append({"error": f"Error confirming booking: {str(e)}. Please try again.", "role": "assistant"})
        return state

    start_time = datetime.fromisoformat(selected_slot.replace('Z', '+00:00'))
    end_time = start_time + timedelta(minutes=30)
    async with httpx.AsyncClient() as client:
        try:
            booking_request = {
                "start_time": start_time.astimezone(pytz.UTC).isoformat(),
                "end_time": end_time.astimezone(pytz.UTC).isoformat(),
                "summary": "Meeting",
                "description": "Booked via TailorTalk"
            }
            response = await client.post(f"{BACKEND_URL}/book", json=booking_request)
            response.raise_for_status()
            state["messages"].append({"content": f"Booking confirmed: {response.json()['message']}", "role": "assistant"})
        except httpx.HTTPStatusError as e:
            state["messages"].append({"error": f"Error booking appointment: {str(e)}", "role": "assistant"})

    return state

async def cancel_booking(state: AgentState) -> AgentState:
    """Handle cancellation of a booking in Asia/Kolkata."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BACKEND_URL}/upcoming-events")
            response.raise_for_status()
            state["events"] = response.json()["events"]
        except httpx.HTTPStatusError as e:
            state["messages"].append({"error": f"Error fetching upcoming events: {str(e)}", "role": "assistant"})
            return state
    
    message = state["messages"][-1]["content"].strip().lower()
    events = state["events"]
    if not events:
        state["messages"].append({"content": "No upcoming bookings found.", "role": "assistant"})
        return state
    
    tz = pytz.timezone("Asia/Kolkata")
    event_list = "\n".join([
        f"{i+1}. {e['summary']} at {datetime.fromisoformat(e['start']['dateTime'].replace('Z', '+00:00')).astimezone(tz).strftime('%B %d, %Y, %I:%M %p')}"
        for i, e in enumerate(events)
    ])
    if "select" not in message and not any(str(i+1) in message for i in range(len(events))):
        state["messages"].append({"content": f"Please select an event to cancel:\n{event_list}", "role": "assistant"})
        return state
    
    selected = None
    for i in range(len(events)):
        if str(i+1) in message:
            selected = i
            break
    if selected is None:
        state["messages"].append({"content": f"Please specify the event number to cancel:\n{event_list}", "role": "assistant"})
        return state
    
    async with httpx.AsyncClient() as client:
        try:
            event_id = events[selected]['id']
            event_time = datetime.fromisoformat(events[selected]['start']['dateTime'].replace('Z', '+00:00')).astimezone(tz).strftime("%B %d, %Y, %I:%M %p")
            response = await client.delete(f"{BACKEND_URL}/cancel/{event_id}")
            response.raise_for_status()
            state["messages"].append({"content": f"Cancelled booking: {events[selected]['summary']} at {event_time}", "role": "assistant"})
        except httpx.HTTPStatusError as e:
            state["messages"].append({"error": f"Error cancelling booking: {str(e)}", "role": "assistant"})
    return state

def route(state: AgentState) -> str:
    """Route to the appropriate node based on intent."""
    intent = state["intent"]
    if intent == "book_appointment":
        return "check_availability"
    elif intent == "check_availability":
        return "check_availability"
    elif intent == "confirm_booking":
        return "confirm_booking"
    elif intent == "cancel":
        return "cancel"
    else:
        return "suggest_slots"

# Build the LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("identify_intent", identify_intent)
workflow.add_node("check_availability", check_availability_node)
workflow.add_node("suggest_slots", suggest_slots)
workflow.add_node("confirm_booking", confirm_booking)
workflow.add_node("cancel", cancel_booking)

workflow.set_entry_point("identify_intent")
workflow.add_conditional_edges(
    "identify_intent",
    route,
    {
        "check_availability": "check_availability",
        "suggest_slots": "suggest_slots",
        "confirm_booking": "confirm_booking",
        "cancel": "cancel"
    }
)
workflow.add_edge("check_availability", "suggest_slots")
workflow.add_edge("suggest_slots", END)
workflow.add_edge("confirm_booking", END)
workflow.add_edge("cancel", END)

graph = workflow.compile()

# --- Streamlit Frontend with WhatsApp-Style Layout ---
st.title("TailorTalk - Appointment Booking Agent")
st.write("Welcome to TailorTalk! I can assist with your appointments in India. Try saying: 'Book a meeting tomorrow' or 'Show me free slots this Friday.'")

# Initialize session state with explicit roles
if "messages" not in st.session_state:
    st.session_state.messages = [{"content": "Hello! How can I assist you today?", "role": "assistant"}]

# Container for chat with WhatsApp-style flow
chat_container = st.container()

# Display chat history sequentially
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.chat_message("assistant").write(f"**AI:** {message['content']}", unsafe_allow_html=True)
        else:
            st.chat_message("user").write(f"**You:** {message['content']}", unsafe_allow_html=True)

# Chat input and processing
async def main():
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"content": prompt, "role": "user"})
        with chat_container:
            st.chat_message("user").write(f"**You:** {prompt}", unsafe_allow_html=True)
        
        with st.spinner("Processing your request..."):
            try:
                state = {
                    "messages": [{"content": msg["content"], "role": msg["role"]} for msg in st.session_state.messages],
                    "intent": "",
                    "suggested_slots": [],
                    "confirmed_slot": "",
                    "time_zone": "Asia/Kolkata",
                    "events": []
                }
                result = await graph.ainvoke(state)
                new_message = result["messages"][-1]
                # Ensure the new message is always assigned the assistant role
                new_message["role"] = "assistant"
                st.session_state.messages.append(new_message)
                with chat_container:
                    st.chat_message("assistant").write(f"**AI:** {new_message['content']}", unsafe_allow_html=True)
            except Exception as e:
                error_msg = {"content": f"Sorry, something went wrong: {str(e)}. Please try again.", "role": "assistant"}
                st.session_state.messages.append(error_msg)
                with chat_container:
                    st.chat_message("assistant").write(f"**AI:** {error_msg['content']}", unsafe_allow_html=True)

if __name__ == "__main__":
    asyncio.run(main())
