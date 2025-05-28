import json
import os


def save_model(data, path="../store", name="model"):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{name}.json")
    with open(full_path, 'w') as f:
        json.dump(data, f)


def load_model(path="../store/model.json"):
    with open(path, 'r') as f:
        data = f.read()
        return json.loads(data)
