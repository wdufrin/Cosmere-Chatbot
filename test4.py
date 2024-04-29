from typing import Optional
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Tool
import vertexai.preview.generative_models as generative_models


def multiturn_generate_content(model: GenerativeModel, user_input: str) -> str:
  chat = model.start_chat()
  return chat.send_message(
      [user_input],
      generation_config=generation_config,
      safety_settings=safety_settings
  )


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

def create_session():
    chat_model = model
    chat = model.start_chat()
    return chat


def run_chat():
    chat_model = create_session()
    print(f"Chat Session created")
    
    while True:
        user_input = input("Input: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        content = multiturn_generate_content(model, user_input)
        print(f"AI: {content.text}")

if __name__ == '__main__':
    vertexai.init(project="ancient-sandbox-322523", location="us-central1")
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
    run_chat()
