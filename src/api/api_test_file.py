import requests

url = 'http://127.0.0.1:5000/predict'

json_data = {'Username': ['bennygui','Tonydorrf'],
             'num_of_games': [5,5]}
response = requests.post(url=url, json=json_data)
for item in response.json():
    print(item, 'recommended games: ', ', '.join(response.json().get(item)))

