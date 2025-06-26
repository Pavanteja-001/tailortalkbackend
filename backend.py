from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from datetime import datetime, timedelta
import pytz
from pydantic import BaseModel
import os
import logging
import webbrowser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware to allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Calendar API setup
SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDENTIALS_FILE = 'credentials.json'

class BookingRequest(BaseModel):
    start_time: str
    end_time: str
    summary: str
    description: str

def get_calendar_service():
    try:
        creds = None
        if os.path.exists('token.json'):
            logger.info("Loading credentials from token.json")
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired token")
                creds.refresh(Request())
            else:
                logger.info("Initiating OAuth flow")
                if not os.path.exists(CREDENTIALS_FILE):
                    logger.error(f"Credentials file {CREDENTIALS_FILE} not found")
                    raise FileNotFoundError(f"Credentials file {CREDENTIALS_FILE} not found")
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                flow.redirect_uri = 'http://localhost:8000/oauth2callback'
                auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
                logger.info(f"Please visit this URL to authorize the application: {auth_url}")
                print(f"Please visit this URL to authorize the application: {auth_url}")
                webbrowser.open(auth_url)
                raise HTTPException(
                    status_code=200,
                    detail=f"Please complete authorization by visiting: {auth_url}"
                )
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        service = build('calendar', 'v3', credentials=creds, cache_discovery=False)
        logger.info("Google Calendar service initialized")
        return service
    except Exception as e:
        logger.error(f"Error in get_calendar_service: {str(e)}")
        raise

@app.get("/availability")
async def get_availability(start: str = Query(...), end: str = Query(...)):
    try:
        logger.info(f"Received availability request: start={start}, end={end}")
        # Normalize datetime strings to handle potential malformed input
        start = start.replace(' ', '+').replace(' 00:00', '+00:00').strip()
        end = end.replace(' ', '+').replace(' 00:00', '+00:00').strip()
        # Parse datetime strings
        try:
            start_time = datetime.fromisoformat(start)
            end_time = datetime.fromisoformat(end)
        except ValueError as e:
            logger.error(f"Invalid datetime format: {str(e)}, start={start}, end={end}")
            raise HTTPException(status_code=400, detail=f"Invalid datetime format: {str(e)}")
        # Ensure datetimes are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=pytz.UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=pytz.UTC)
        logger.info(f"Parsed start_time: {start_time.isoformat()}, end_time: {end_time.isoformat()}")
        # Validate time range
        if start_time >= end_time:
            raise HTTPException(status_code=400, detail="start_time must be before end_time")
        # Convert to Asia/Kolkata for business hours filtering
        tz = pytz.timezone('Asia/Kolkata')
        start_ist = start_time.astimezone(tz)
        end_ist = end_time.astimezone(tz)
        service = get_calendar_service()
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_time.isoformat(),
            timeMax=end_time.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        logger.info(f"Found {len(events)} events")
        busy_slots = [
            {
                'start': event['start'].get('dateTime'),
                'end': event['end'].get('dateTime')
            }
            for event in events
        ]
        available_slots = []
        current_time = start_time
        while current_time < end_time:
            slot_end = current_time + timedelta(minutes=30)
            current_time_ist = current_time.astimezone(tz)
            # Restrict to business hours (9:00 AM to 6:00 PM IST)
            if 9 <= current_time_ist.hour < 18:
                if not any(
                    datetime.fromisoformat(slot['start'].replace('Z', '+00:00')) < slot_end and
                    datetime.fromisoformat(slot['end'].replace('Z', '+00:00')) > current_time
                    for slot in busy_slots
                ):
                    available_slots.append(current_time.isoformat())
            current_time = slot_end
        logger.info(f"Returning {len(available_slots)} available slots")
        return {"available_slots": available_slots}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in get_availability: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching availability: {str(e)}")

@app.post("/book")
async def book_appointment(booking: BookingRequest):
    try:
        logger.info(f"Booking appointment: {booking}")
        start_time = datetime.fromisoformat(booking.start_time.replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(booking.end_time.replace('Z', '+00:00'))
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=pytz.UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=pytz.UTC)
        if start_time >= end_time:
            raise HTTPException(status_code=400, detail="start_time must be before end_time")
        service = get_calendar_service()
        event = {
            'summary': booking.summary,
            'description': booking.description,
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': 'UTC',
            },
        }
        event = service.events().insert(calendarId='primary', body=event).execute()
        logger.info(f"Event created: {event.get('htmlLink')}")
        return {"message": f"Event created: {event.get('htmlLink')}"}
    except ValueError as e:
        logger.error(f"Invalid datetime format: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {str(e)}")
    except Exception as e:
        logger.error(f"Error in book_appointment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error booking appointment: {str(e)}")

@app.get("/upcoming-events")
async def get_upcoming_events():
    try:
        logger.info("Fetching upcoming events")
        service = get_calendar_service()
        now = datetime.now(pytz.UTC)
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now.isoformat(),
            maxResults=10,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        logger.info(f"Found {len(events)} upcoming events")
        return {"events": events}
    except Exception as e:
        logger.error(f"Error in get_upcoming_events: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching events: {str(e)}")

@app.delete("/cancel/{event_id}")
async def cancel_event(event_id: str):
    try:
        logger.info(f"Cancelling event: {event_id}")
        service = get_calendar_service()
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        logger.info("Event cancelled")
        return {"message": "Event cancelled"}
    except Exception as e:
        logger.error(f"Error in cancel_event: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cancelling event: {str(e)}")

@app.get("/oauth2callback")
async def oauth2callback(code: str = Query(None), state: str = Query(None)):
    try:
        logger.info("Handling OAuth callback")
        if not code:
            logger.error("Authorization code not provided")
            raise HTTPException(status_code=400, detail="Authorization code not provided")
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        flow.redirect_uri = 'http://localhost:8000/oauth2callback'
        flow.fetch_token(code=code)
        creds = flow.credentials
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
        logger.info("OAuth callback successful, token saved")
        return {"message": "Authorization successful. You can close this window and retry the request."}
    except Exception as e:
        logger.error(f"Error in oauth2callback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in OAuth callback: {str(e)}")