import json

with open('saved_sessions/688ffd30-ace0-4a9e-9ceb-b40d3132af6c.json', encoding='utf-8') as f:
    data = json.load(f)

print(f"Messages in file: {len(data.get('messages', []))}")
if data.get('messages'):
    print(f"First message type: {data['messages'][0].get('llm_type')}")
    print(f"First message length: {len(data['messages'][0].get('content', ''))}")
