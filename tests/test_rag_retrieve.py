import pytest

from diagnosis_agent.state import GraphState
from diagnosis_agent.nodes.rag_node import rag_retrieve


@pytest.mark.asyncio
async def test_rag_retrieve():
    state: GraphState = {"question": "What is diabetes?"}

    result = await rag_retrieve(state)

    assert "contexts" in result

    print("Retrieved Contexts:", result["contexts"])
