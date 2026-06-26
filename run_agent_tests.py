import asyncio
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="C:/LLM/.env")

# Ensure scratch directory is in python path
sys.path.append(os.path.dirname(__file__))
from agent_setup import get_agent_config
try:
    import sys
    if sys.platform == "win32":
        raise RuntimeError("Windows requires localharness shim")
    from google.antigravity import Agent
except (ImportError, RuntimeError):
    from google_antigravity_shim import Agent

async def run_tests():
    print("Initializing Google Antigravity Agent...")
    config = get_agent_config()
    
    async with Agent(config) as agent:
        print("\n================ TEST 1: PR Gap Breakdown ================")
        prompt_1 = "What is the overall generation performance and PR gap breakdown for the plant?"
        print(f"User: {prompt_1}")
        
        response_1 = await agent.chat(prompt_1)
        print("Agent Thoughts:")
        async for thought in response_1.thoughts:
            print(thought, end="", flush=True)
        print("\n\nAgent Output:")
        print(await response_1.text())
        
        print("\n================ TEST 2: Inverter String Outliers ================")
        prompt_2 = "Are there any underperforming strings on Inverter B1INV1? If so, generate a warning ticket."
        print(f"User: {prompt_2}")
        
        response_2 = await agent.chat(prompt_2)
        print("Agent Thoughts:")
        async for thought in response_2.thoughts:
            print(thought, end="", flush=True)
        print("\n\nAgent Output:")
        print(await response_2.text())
        
        print("\n================ TEST 3: BESS Health ================")
        prompt_3 = "What is the health and cycle history of battery B1BCT1?"
        print(f"User: {prompt_3}")
        
        response_3 = await agent.chat(prompt_3)
        print("Agent Thoughts:")
        async for thought in response_3.thoughts:
            print(thought, end="", flush=True)
        print("\n\nAgent Output:")
        print(await response_3.text())

if __name__ == "__main__":
    asyncio.run(run_tests())
