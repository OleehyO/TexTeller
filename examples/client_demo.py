import requests

server_url = "http://127.0.0.1:8000/predict"

img_path = "/path/to/your/image"
with open(img_path, "rb") as img:
    files = {"img": img}
    response = requests.post(server_url, files=files)

print(response.text)
