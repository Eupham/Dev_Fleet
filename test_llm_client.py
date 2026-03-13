import asyncio
from orchestrator.llm_client import chat_completion

def main():
    print(chat_completion([{"role": "user", "content": "hi"}]))

if __name__ == "__main__":
    main()
