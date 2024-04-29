from typing import Optional

import vertexai
from vertexai.preview.generative_models import (
    GenerationResponse,
    GenerativeModel,
    grounding,
    Tool,
)

PROJECT_ID = "ancient-sandbox-322523"  
LOCATION = "us-central1" 
data_store_path = "projects/ancient-sandbox-322523/locations/us-central1/collections/default_collection/dataStores/sunlit2_1704293809107"
model = GenerativeModel(model_name="gemini-1.0-pro")

def generate_text_with_grounding(
    PROJECT_ID: str, LOCATION: str, data_store_path: str, message: str
) -> GenerationResponse:
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    data_store_path = "projects/ancient-sandbox-322523/locations/us-central1/collections/default_collection/dataStores/sunlit2_1704293809107"

    # Create Tool for grounding
    tool = Tool.from_retrieval(
        grounding.Retrieval(grounding.VertexAISearch(datastore=data_store_path))
    )

 #   print(f"message:", message)
 #   print(f"tool:", tool)
    
    
    response = model.generate_content(message)
    #response = model.generate_content(message, tools=[tool])
    return response


def create_session():
    model = GenerativeModel(model_name="gemini-1.0-pro")
    chat = model.start_chat()
    return chat

def run_chat():
    chat_model = create_session()
    print(f"Chat Session created")
    
    while True:
        user_input = input("Input: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        content = response(chat_model, user_input)
        print(f"AI: {content}")

def response(chat, message):
    parameters = {
        "candidate_count": 1,
        "temperature": 0,
        "max_output_tokens": 2048,
        "top_p": 1
    }

    result = generate_text_with_grounding(PROJECT_ID,LOCATION,data_store_path,message)
    print("RESULTS: ", result.text)
    return result.text


if __name__ == '__main__':
    run_chat()


