import requests
name = 'hjhj'
# r = requests.get(f'http://127.0.0.1:5000/api/ml_models/model_info/{name}')
r = requests.get('http://127.0.0.1:5000/api/ml_models/models_info')
print(r.json())