# import the required libraries
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
import pickle
import os.path
import base64
import email
import re

def strip_tags(text):
    text = re.sub(r'<[^>]*>', '', text, flags=re.DOTALL)  # Remove HTML tags
    text = re.sub(r'\{[^{}]*?\}', '', text, flags=re.DOTALL)  # Remove content between {}
    return text

# Define the SCOPES. If modifying it, delete the token.pickle file.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def getEmails():
    # Variable creds will store the user access token.
    # If no valid token found, we will create one.
    creds = None
    # The file token.pickle contains the user access token.
    # Check if it exists
    if os.path.exists('token.pickle'):
        # Read the token from the file and store it in the variable creds
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If credentials are not available or are invalid, ask the user to log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the access token in token.pickle file for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Connect to the Gmail API
    service = build('gmail', 'v1', credentials=creds)

    # request a list of all the messages
    result = service.users().messages().list(userId='me').execute()

    # We can also pass maxResults to get any number of emails. Like this:
    # result = service.users().messages().list(maxResults=200, userId='me').execute()
    messages = result.get('messages')

    # print(messages)
    # messages is a list of dictionaries where each dictionary contains a message id.
    # iterate through all the messages
    for msg in messages:


        # Get the message from its id
        txt = service.users().messages().get(userId='me', id=msg['id']).execute()

        # Use try-except to avoid any Errors
        try:
            # Get value of 'payload' from dictionary 'txt'
            payload = txt['payload']
            headers = payload['headers']

            # Look for Subject and Sender Email in the headers
            for d in headers:
                if d['name'] == 'Subject':
                    subject = d['value']
                if d['name'] == 'From':
                    sender = d['value']

            # Get the message body
            message_body = ''
            if 'data' in payload['body']:
                data = payload['body']['data']
                message_body = base64.urlsafe_b64decode(data).decode('utf-8')
            elif 'parts' in payload:
                parts = payload['parts']
                for part in parts:
                    if part['mimeType'] == 'text/html':
                        data = part['body']['data']
                        message_body = base64.urlsafe_b64decode(data).decode('utf-8')
                        break

            # Remove HTML tags and content between curly braces, handling multi-line cases
            message_body = strip_tags(message_body)
            message_body = '\n'.join(line.strip() for line in message_body.splitlines() if line.strip())

            # Printing the subject, sender's email, and message
            print("Subject: ", subject)
            print("From: ", sender)
            print("msg['id']: ", msg['id'])
            print("Message: ", message_body)
            print('\n')
            break
        except HttpError as error:
            print(f'An error occurred: {error}')

getEmails()