
import os
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import GroundingSource
from google.cloud import aiplatform
from vertexai.language_models import ChatModel, InputOutputTextPair

PROJECT_ID = "ancient-sandbox-322523"  
LOCATION = "us-central1" 

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = ChatModel.from_pretrained("chat-bison")

def create_session():
    chat_model = vertexai.language_models.ChatModel.from_pretrained("chat-bison@001")
    chat = chat_model.start_chat()
    return chat

def response(chat, message):
    parameters = {
        "candidate_count": 1,
        "temperature": 0,
        "max_output_tokens": 1024,
        "top_p": 1
    }

    grounding_source=GroundingSource.VertexAISearch(data_store_id="sunlit2_1704293809107", location="global", project="180054373655")
    result = chat.send_message(message, **parameters,grounding_source=grounding_source)
    return result.text

def run_chat():
    chat_model = create_session()
    print(f"Chat Session created")
    
    while True:
        user_input = input("Input: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        content = response(chat_model, user_input)
        print(f"AI: {content}")

if __name__ == '__main__':
    run_chat()