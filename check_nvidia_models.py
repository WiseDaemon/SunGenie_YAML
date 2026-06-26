import urllib.request
import json

url = "https://integrate.api.nvidia.com/v1/models"
headers = {
    "Authorization": "Bearer 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"
}

req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=15) as response:
        models = json.loads(response.read().decode("utf-8"))
        print("Models count:", len(models.get("data", [])))
        for m in models.get("data", [])[:10]:
            print("-", m.get("id"))
except Exception as e:
    print("Error:", e)
