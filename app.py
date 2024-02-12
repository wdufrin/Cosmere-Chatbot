from flask import Flask, render_template, request, jsonify
import vertexai
from vertexai.language_models import ChatModel
from vertexai.language_models import GroundingSource
import os
from vertexai.language_models import ChatModel, InputOutputTextPair


app = Flask(__name__)
PROJECT_ID = "ancient-sandbox-322523"  
LOCATION = "us-central1"  

vertexai.init(project=PROJECT_ID, location=LOCATION)

def create_session():
    chat_model = vertexai.language_models.ChatModel.from_pretrained("chat-bison@001")
    chat = chat_model.start_chat()
    return chat

def response(chat, message):
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 1024,
        "top_p": 1
    }
    grounding_source=GroundingSource.VertexAISearch(data_store_id="sunlit2_1704293809107", location="global", project="180054373655")
    result = chat.send_message(message, **parameters,grounding_source=grounding_source)
    return result.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/palm2', methods=['GET', 'POST'])
def vertex_palm():
    user_input = ""
    if request.method == 'GET':
        user_input = request.args.get('user_input')
    else:
        user_input = request.form['user_input']
    chat_model = create_session()
    content = response(chat_model,user_input)
    return jsonify(content=content)

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')