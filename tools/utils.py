import json, os


def save_model(data, path="../store/model.json"):
    data = json.dumps(data)
    os.makedirs(os.path.join("../store"), exist_ok=True)
    with open(path, 'w') as f:
        f.write(data)


def load_model(path="../store/weights.json"):
    with open(path, 'r') as f:
        data = f.read()
        return json.loads(data)
