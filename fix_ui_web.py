import re
content = open("ui/web.py").read()

new_content = content.replace(
"""            step_name = update["step"]
            state_snapshot = update["state_snapshot"]
            graphs_dict = update["graphs"]

            # Display the execution step
            async with cl.Step(name=step_name) as step:""",
"""            step_name = update["step"]
            state_snapshot = update["state_snapshot"]
            graphs_dict = update["graphs"]

            if step_name == "keep-alive":
                # Just ignore keep-alive to keep connection open
                continue

            # Display the execution step
            async with cl.Step(name=step_name) as step:"""
)
open("ui/web.py", "w").write(new_content)
