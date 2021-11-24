import requests
import json

class Translator:
    def __init__(self, base_url="http://localhost:8000/translate/parallel"):
        self.base_url = base_url

    def build_url(self, source_language, target_language):
        return self.base_url + f"/{target_language}?srcLang={source_language}"

    def translate(self, text, source_language, target_language, **kwargs):
        url = self.build_url(source_language, target_language)
        data = {'text': text}
        x = requests.post(url, json=data)
        transl = json.loads(x.text)
        if isinstance(transl, list):
            return {"translatedText": transl}
        else:
            return None
