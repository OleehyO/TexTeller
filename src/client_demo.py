import requests

url = "http://127.0.0.1:8000/predict"

img_path = "/your/image/path/"
with open(img_path, 'rb') as img:
    files = {'img': img}
    response = requests.post(url, files=files)

# data = {"img_path": img_path}

# response = requests.post(url, json=data)

print(response.text)
