import requests

rec_server_url = "http://127.0.0.1:8000/frec"
det_server_url = "http://127.0.0.1:8000/fdet"

img_path = "/your/image/path/"
with open(img_path, 'rb') as img:
    files = {'img': img}
    response = requests.post(rec_server_url, files=files)
    # response = requests.post(det_server_url, files=files)

print(response.text)
