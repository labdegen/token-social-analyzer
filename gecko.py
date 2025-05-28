import requests

url = "https://api.coingecko.com/api/v3/search/trending"
headers = {"accept": "application/json"}
response = requests.get(url, headers=headers, timeout=10)

if response.status_code == 200:
    print("Success:", response.json())
else:
    print("Failed:", response.status_code, response.text)