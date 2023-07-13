import os
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify
import datetime
import openai
from threading import Lock

from src import utils

app = Flask(__name__)

# Store conversations in memory
conversations = {}
conversations_lock = Lock()


@app.route('/', methods=['GET'])
def home():
    return 'Welcome to the chatbot API!'


@app.route('/showConversations', methods=['GET'])
def showConversations():
    return str(conversations)


@app.route('/chat', methods=['POST'])
def chat():
    # Get the JSON payload from the request
    json_data = request.get_json()

    if not json_data:
        return jsonify({'error': 'Invalid JSON payload'})

    # Extract the user ID and query from the JSON payload
    user_id = json_data.get('id')
    user_query = json_data.get('query')

    with conversations_lock:
        if user_id in conversations:
            conversation = conversations[user_id]
        else:
            conversation = []
            conversations[user_id] = conversation

        # Make the API call to generate a response
        bot_response = utils.get_completion(user_query)

        # Append the user query, bot response, and timestamp to the conversation
        conversation.append({
            'user_query': user_query,
            'bot_response': bot_response,
            'timestamp': str(datetime.datetime.now())
        })

    # Return the response as JSON
    return jsonify({
        'id': user_id,
        'response': bot_response,
        'timestamp': conversation[-1]['timestamp']
    })


if __name__ == '__main__':
    # Set up your OpenAI API key
    load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ['OPENAI_API_KEY']

    app.run()
