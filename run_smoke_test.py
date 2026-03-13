import sys
import json
import modal

def main():
    prompt = "Write a hello world function in Python"
    if len(sys.argv) > 1:
        prompt = sys.argv[1]

    # Connect to the deployed "dev_fleet" app
    func = modal.Function.from_name("dev_fleet", "run_agent")

    print(f"Running smoke test on deployed dev_fleet with prompt: '{prompt}'")
    result = func.remote(prompt)
    print("Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
