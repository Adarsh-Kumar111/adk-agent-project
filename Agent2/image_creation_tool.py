import os
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from dotenv import load_dotenv
from typing import Union
from google.adk.tools import ToolContext

async def create_image(prompt: str, tool_context: ToolContext) -> Union[bytes, str]:

    try:
        dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
        load_dotenv(dotenv_path=dotenv_path)

        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        model_name = os.getenv("IMAGEN_MODEL")

        vertexai.init(project=project_id, location=location)

        model = ImageGenerationModel.from_pretrained(model_name)

        images = model.generate_images(
            prompt=prompt,
            number_of_images=1
        )

        for image in images:
            image_bytes = image._image_bytes

            artifact_name = "generated_image.png"

            await tool_context.save_artifact(
                artifact_name,
                image_bytes,
                mime_type="image/png"
            )

            return {
                "status": "success",
                "artifact_name": artifact_name
            }

    except Exception as e:
        return f"Error generating image: {e}"