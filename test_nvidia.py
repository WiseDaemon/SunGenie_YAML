import json
import urllib.request
import urllib.error

url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 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"
}

data = {
    "model": "nvidia/nemotron-3-ultra-550b-a55b",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 1,
    "top_p": 0.95,
    "max_tokens": 100,
    "chat_template_kwargs": {"enable_thinking": True},
    "reasoning_budget": 1024
}

req = urllib.request.Request(
    url,
    data=json.dumps(data).encode("utf-8"),
    headers=headers,
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=30) as response:
        print("Success:")
        print(response.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    print(f"HTTPError: {e.code} {e.reason}")
    print(e.read().decode("utf-8"))
except Exception as e:
    print(f"Error: {e}")
