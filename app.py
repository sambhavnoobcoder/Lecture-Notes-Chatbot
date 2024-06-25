import requests
import numpy as np

# Fetch lecture notes and model architectures
def fetch_lecture_notes():
    lecture_urls = [
        "https://stanford-cs324.github.io/winter2022/lectures/introduction/",
        "https://stanford-cs324.github.io/winter2022/lectures/capabilities/",
        "https://stanford-cs324.github.io/winter2022/lectures/data/",
        "https://stanford-cs324.github.io/winter2022/lectures/modeling/"
    ]
    lecture_texts = []
    for url in lecture_urls:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Fetched content from {url}")
            lecture_texts.append((extract_text_from_html(response.text), url))
        else:
            print(f"Failed to fetch content from {url}, status code: {response.status_code}")
    return lecture_texts


def fetch_model_architectures():
    url = "https://github.com/Hannibal046/Awesome-LLM#milestone-papers"
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Fetched model architectures, status code: {response.status_code}")
        return extract_text_from_html(response.text), url
    else:
        print(f"Failed to fetch model architectures, status code: {response.status_code}")
        return "", url
