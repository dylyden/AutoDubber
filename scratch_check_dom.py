import re
with open('app.js', 'r', encoding='utf-8') as f:
    js_text = f.read()
with open('index.html', 'r', encoding='utf-8') as f:
    html_text = f.read()

missing = []
ids = re.findall(r"getElementById\(['\"](.*?)['\"]\)", js_text)
for el_id in ids:
    if f'id="{el_id}"' not in html_text and f"id='{el_id}'" not in html_text:
        missing.append(el_id)
print("MISSING:", set(missing))
