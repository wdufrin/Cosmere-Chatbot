from flask import Flask, render_template, request, jsonify
import vertexai
import base64
from typing import Optional
from vertexai.generative_models import GenerativeModel, Part, Tool
import vertexai.preview.generative_models as generative_models
from vertexai.language_models import GroundingSource



# Create a Flask app
app = Flask(__name__)

# Set the project ID and location for Vertex AI
PROJECT_ID = "ancient-sandbox-322523"  
LOCATION = "us-central1"  

# Initialize the Vertex AI client library
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Define a function to create a chat session
def create_session():
    # Create a list of tools to use for grounding
    tools = [
        Tool.from_retrieval(
            # Create a retrieval tool that uses the Vertex AI Search API
            retrieval=generative_models.grounding.Retrieval(
                source=generative_models.grounding.VertexAISearch(datastore="projects/180054373655/locations/global/collections/default_collection/dataStores/sunlit2_1704293809107"),
                disable_attribution=False,
            )
        ),
    ]
    parameters = {
    "max_output_tokens": 8192,
    "temperature": .5,
    "top_p": 0.95
    }
    # Create a chat model with the specified tools
    chat_model = GenerativeModel("gemini-1.5-pro-preview-0409", tools=tools)
    # Start a chat session with the chat model
    chat = chat_model.start_chat()
    return chat

# Define a function to send a message to the chat session and get a response
def response(chat, message):
    # Set the grounding source for the message
    #grounding_source=GroundingSource.VertexAISearch(data_store_id="sunlit2_1704293809107", location="global", project="180054373655")
    # Send the message to the chat session and get a response
    result = chat.send_message(message)
    return result.text

# Define the main route for the app
@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

# Define the route for handling POST requests to the /chat endpoint
@app.route('/chat', methods=['GET', 'POST'])
def vertex_palm():
    # Get the user input from the request
    user_input = ""
    if request.method == 'GET':
        user_input = request.args.get('user_input')
    else:
        user_input = request.form['user_input']

    # Create a chat session
    chat_model = create_session()

    # Get a response from the chat session
    content = response(chat_model, user_input)

    # Return the response as JSON
    return jsonify(content=content)


# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
