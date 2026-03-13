import re

content = open("orchestrator/agent_loop.py").read()

new_content = content.replace(
"""    # Stream over nodes for real-time visibility
    for s in app.stream(initial_state, config=config):
        # s is a dict with the key being the node name and the value being the new state
        node_name = list(s.keys())[0]
        state_data = s[node_name]

        # Load memory to get the current graph state
        memory = TriGraphMemory.load()
        yield {
            "step": node_name,
            "state_snapshot": state_data,
            "graphs": memory.to_dict()
        }""",
"""    # Stream over nodes for real-time visibility
    import threading
    import queue
    import time

    update_queue = queue.Queue()

    def run_graph():
        for s in app.stream(initial_state, config=config):
            update_queue.put(("update", s))
        update_queue.put(("done", None))

    t = threading.Thread(target=run_graph)
    t.start()

    while True:
        try:
            msg_type, data = update_queue.get(timeout=30.0)
            if msg_type == "done":
                break

            node_name = list(data.keys())[0]
            state_data = data[node_name]

            # Load memory to get the current graph state
            memory = TriGraphMemory.load()
            yield {
                "step": node_name,
                "state_snapshot": state_data,
                "graphs": memory.to_dict()
            }
        except queue.Empty:
            # Yield a keep-alive message every 30 seconds
            memory = TriGraphMemory.load()
            yield {
                "step": "keep-alive",
                "state_snapshot": {"messages": ["Waiting for tasks to complete..."]},
                "graphs": memory.to_dict()
            }
"""
)

open("orchestrator/agent_loop.py", "w").write(new_content)
