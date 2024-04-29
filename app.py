from flask import Flask, request, jsonify
from typing import Optional
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Tool
import vertexai.preview.generative_models as generative_models

app = Flask(__name__)
PROJECT_ID = "ancient-sandbox-322523"  
LOCATION = "us-central1"  

generation_config = {
    "max_output_tokens": 8192,
    "temperature": .5,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

def create_model():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    tools = [
        Tool.from_retrieval(
            retrieval=generative_models.grounding.Retrieval(
                source=generative_models.grounding.VertexAISearch(datastore="projects/180054373655/locations/global/collections/default_collection/dataStores/sunlit2_1704293809107"),
                disable_attribution=False,
            )
        ),
    ]
    model = GenerativeModel(
        "gemini-1.5-pro-preview-0409",
        tools=tools,
        system_instruction=["""You are a fun and whitty companion.  You always tell the truth and are not afraid to say you do not know the answer to something."""]
    )
    return model

model = create_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    user_input = request.json['message']
    if request.method == 'GET':
        user_input = request.args.get('message')
    else:
        user_input = request.json['message']
    content = multiturn_generate_content(model, user_input)
    return jsonify({'message': content.text})

def multiturn_generate_content(model: GenerativeModel, user_input: str) -> str:
  chat = model.start_chat()
  return chat.send_message(
      [user_input],
      generation_config=generation_config,
      safety_settings=safety_settings
  )

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
