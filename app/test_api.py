import requests
url = "http://localhost:9696/flower-classification"

data = {'url':'https://github.com/PriyaVellanki/flower_classification/raw/main/data/11124324295_503f3a0804.jpg'}

result = requests.post(url, json=data).json()
print(result)