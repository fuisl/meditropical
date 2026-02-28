import asyncio

from diagnosis_agent.state import GraphState
from diagnosis_agent.workflow import build_workflow


async def main():
    workflow = build_workflow()

    initial_state: GraphState = {
        "question": "What is Malaria and how is it treated?",
        "contexts": None,
        "answer": None,
        "reasoning": None,
        "evaluation": None,
    }

    # invoke the compiled workflow. LangGraph returns the final state after executing nodes.
    result = await workflow.ainvoke(initial_state)

    print("=== Final state ===")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
