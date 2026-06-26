import asyncio
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="C:/LLM/.env")

sys.path.append(os.path.dirname(__file__))
from agent_setup import get_agent_config
from google_antigravity_shim import Agent

async def run_tests():
    print("Initializing Google Antigravity Agent for custom testing...")
    config = get_agent_config()
    
    async with Agent(config) as agent:
        print("\n================ TEST 1: List Assets by Type ================")
        prompt_1 = "list down all the assets by type"
        print(f"User: {prompt_1}")
        response_1 = await agent.chat(prompt_1)
        print("\nAgent Output:")
        print(await response_1.text())
        
        print("\n================ TEST 2: Insights on MFM 12 ================")
        prompt_2 = "insights on MFM 12"
        print(f"User: {prompt_2}")
        response_2 = await agent.chat(prompt_2)
        print(f"Thoughts: {response_2._thoughts}")
        print("\nAgent Output:")
        print(await response_2.text())
        print(f"Chart generated: {response_2.chart is not None}")
        if response_2.chart:
            print(f"Chart details: Type: {response_2.chart['type']}, Title: {response_2.chart['title']}")
            
        print("\n================ TEST 3: Compare B2_BCT3 and B2_BCT1 ================")
        prompt_3 = "compare B2_BCT3 and B2_BCT1"
        print(f"User: {prompt_3}")
        response_3 = await agent.chat(prompt_3)
        print(f"Thoughts: {response_3._thoughts}")
        print("\nAgent Output:")
        print(await response_3.text())
        print(f"Chart generated: {response_3.chart is not None}")
        if response_3.chart:
            print(f"Chart details: Type: {response_3.chart['type']}, Title: {response_3.chart['title']}")

        print("\n================ TEST 4: Compare Meters ================")
        prompt_4 = "compare meters"
        print(f"User: {prompt_4}")
        response_4 = await agent.chat(prompt_4)
        print(f"Thoughts: {response_4._thoughts}")
        print("\nAgent Output:")
        print(await response_4.text())
        print(f"Chart generated: {response_4.chart is not None}")
        if response_4.chart:
            print(f"Chart details: Type: {response_4.chart['type']}, Title: {response_4.chart['title']}")

if __name__ == "__main__":
    asyncio.run(run_tests())
