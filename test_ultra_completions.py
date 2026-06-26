import urllib.request
import json

url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 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"
}

data = {
    "model": "nvidia/nemotron-3-ultra-550b-a55b",
    "messages": [{"role": "user", "content": "Hello! Respond in one sentence."}],
    "temperature": 0.5,
    "max_tokens": 100
}

req = urllib.request.Request(
    url,
    data=json.dumps(data).encode("utf-8"),
    headers=headers,
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=15) as response:
        res = json.loads(response.read().decode("utf-8"))
        print("Success!")
        print(res["choices"][0]["message"]["content"])
except Exception as e:
    print("Error:", e)
