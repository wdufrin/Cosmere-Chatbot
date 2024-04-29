from typing import Optional

#import os
import vertexai
from vertexai.preview.generative_models import (
    GenerationResponse,
    GenerativeModel,
    grounding,
    Tool,
)

#os.environ["ancient-sandbox-322523"] = "ancient-sandbox-322523"


def generate_text_with_grounding(
    project_id: str, location: str, data_store_path: Optional[str] = None
) -> GenerationResponse:
    # Initialize Vertex AI
    vertexai.init(project="ancient-sandbox-322523", location="us-central1")
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 2048,
        "temperature": 0,
        "top_p": 1
    }

    # Load the model
    model = GenerativeModel(model_name="gemini-1.0-pro-001")

    # Create Tool for grounding
    if data_store_path:
        # Use Vertex AI Search data store
        # Format: projects/{project_id}/locations/{location}/collections/default_collection/dataStores/{data_store_id}
        tool = Tool.from_retrieval(
            grounding.Retrieval(grounding.VertexAISearch(datastore="projects/180054373655/locations/global/collections/default_collection/dataStores/sunlit2_1704293809107"))
        )
    else:
        # Use Google Search for grounding (Private Preview)
        tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())

    prompt = "What are the price, available colors, and storage size options of a Pixel Tablet?"
    response = model.generate_content(prompt, tools=[tool])

    print(response)