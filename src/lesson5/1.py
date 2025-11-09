import requests

url = "https://api.siliconflow.cn/v1/embeddings"

payload = {
    "model": "BAAI/bge-m3",
    "input": "Silicon flow embedding online: fast, affordable, and high-quality embedding services. come try it out!"
}
headers = {
    "Authorization": "Bearer sk-ljuovztdpgqafjnufoxhqbuzfzyvctualkzjxekycvpywxpx",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())