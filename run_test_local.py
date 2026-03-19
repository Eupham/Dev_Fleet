import modal

if __name__ == "__main__":
    func = modal.Function.from_name("dev_fleet", "test_tool")
    print("Testing Playwright and UI over deployed MCP server...")
    func.remote()
    print("Done!")
