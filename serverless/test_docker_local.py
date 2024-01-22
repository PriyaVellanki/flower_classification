import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url':'https://github.com/PriyaVellanki/flower_classification/raw/main/data/11124324295_503f3a0804.jpg'}

result = requests.post(url, json=data).json()
print(result)