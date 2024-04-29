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
    PROJECT_ID: str, LOCATION: str, data_store_path: Optional[str] = None
) -> GenerationResponse:
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Create Tool for grounding
    tool = Tool.from_retrieval(grounding.Retrieval(grounding.VertexAISearch(datastore=data_store_path)))
    
prompt = "Who is Tress?"
response = model.generate_content(prompt,tool)

response = generate_text_with_grounding(PROJECT_ID,LOCATION,data_store_path)
print(prompt)
print(response)