import requests

# 服务的 URL
url = "http://127.0.0.1:9900/predict"

# 替换成你要预测的图像的路径
img_path = "/home/lhy/code/TeXify/src/7.png"

# 构造请求数据
data = {"img_path": img_path}

# 发送 POST 请求
response = requests.post(url, json=data)

# 打印响应
print(response.text)
