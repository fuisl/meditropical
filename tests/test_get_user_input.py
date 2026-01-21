import pytest

from diagnosis_agent.state import GraphState
from diagnosis_agent.nodes.input_node import get_user_input


@pytest.mark.asyncio
async def test_get_user_input_with_images():
    state: GraphState = {
        "question": "What disease is shown in this image?",
        "images": [
            {
                "type": "url",
                "value": "https://example.com/image.png",
            },
            {
                "type": "path",
                "value": "data/xray_01.png",
            },
        ],
        "contexts": None,
        "answer": None,
        "reasoning": None,
        "evaluation": None,
    }

    result = await get_user_input(state)

    assert result["question"] == "What disease is shown in this image?"
    assert isinstance(result["images"], list)
    assert len(result["images"]) == 2
    assert result["images"][0]["type"] == "url"
