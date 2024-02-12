import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import GroundingSource

vertexai.init(project="ancient-sandbox-322523", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0,
    "top_p": 1
}
grounding_source = GroundingSource.VertexAISearch(data_store_id="sunlit2_1704293809107", location="global", project="180054373655")
model = TextGenerationModel.from_pretrained("text-bison")
response = model.predict(
    """what's up?""",
    **parameters,
    grounding_source=grounding_source
)
print(f"Response from Model: {response.text}")