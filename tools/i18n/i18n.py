import json
import locale
import os

I18N_JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'locale')

def load_language_list(language):
    with open(os.path.join(I18N_JSON_DIR, f"{language}.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def scan_language_list():
    return [
        name.split('.')[0]
        for name in os.listdir(I18N_JSON_DIR)
        if name.endswith(".json")
    ]

class I18nAuto:
    def __init__(self, language=None):
        if language in (None, "Auto"):
            language = locale.getdefaultlocale()[0] or "en_US"
        path = os.path.join(I18N_JSON_DIR, f"{language}.json")
        if not os.path.exists(path):
            language = "en_US"
            path = os.path.join(I18N_JSON_DIR, f"{language}.json")
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return f"Use Language: {self.language}"

if __name__ == "__main__":
    i18n = I18nAuto(language='en_US')
    print(i18n)
