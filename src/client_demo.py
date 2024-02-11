import requests

url = "http://127.0.0.1:8000/predict"

img_path = "/your/image/path/"

data = {"img_path": img_path}

response = requests.post(url, json=data)

print(response.text)
