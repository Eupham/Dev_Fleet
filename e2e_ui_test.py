import asyncio
import os
import modal

async def run_test():
    func = modal.Function.from_name("dev_fleet", "test_tool")

    print("Testing basic Playwright screenshot functionality...")
    await func.remote.aio("playwright_screenshot")

    print("Executing custom prompt automation script in the remote MCP sandbox...")
    # Now we write a custom python script and send it to the `run_code` MCP tool
    # The script will use Playwright to drive the Chainlit UI.
    script = """
import asyncio
from playwright.async_api import async_playwright
import os

async def interact():
    # The UI URL matches the same domain pattern
    # Use the expected UI url based on Modal environment
    ui_url = "https://ezllm--dev-fleet-ui.modal.run/"

    print(f"Connecting to UI at {ui_url} ...")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        await page.goto(ui_url, timeout=30000)

        # Wait for Chainlit chat input to become available
        await page.wait_for_selector('textarea[id="chat-input"]', timeout=30000)

        prompt = "Research a random domain online. Build an app related to your research. Test it till it runs clean."
        print(f"Submitting prompt: {prompt}")
        await page.fill('textarea[id="chat-input"]', prompt)
        await page.press('textarea[id="chat-input"]', 'Enter')

        # In a real environment, wait for completion.
        # This will take minutes, so we just screenshot the starting execution state for verification
        print("Waiting 30 seconds for agent execution to kick off...")
        await page.wait_for_timeout(30000)

        await page.screenshot(path="/workspace/chainlit_agent_started.png", full_page=True)
        print("Success! Automation completed and screenshot saved to /workspace/chainlit_agent_started.png")

        await browser.close()

asyncio.run(interact())
"""

    # We will use the execute_python tool inside the MCP sandbox via another remote call to `run_code`
    # Wait, `test_tool` only calls the tool passed as argument, but we need to pass arguments!
    # I can just create another Modal function `test_ui_interaction` in `mcp_server.py` to do this correctly,
    # but the prompt didn't ask for a test_tool specifically. Let's just execute `modal.Function.from_name`
    # for the `test_tool` and I'll modify `test_tool` to also run the UI automation if passed "playwright_automation".
    pass

if __name__ == "__main__":
    asyncio.run(run_test())
