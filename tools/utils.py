import json, os


def save_model(data, path="../store/weights.json"):
    data = json.dumps(data)
    os.makedirs(os.path.join("../store"), exist_ok=True)
    with open(path, 'w') as f:
        f.write(data)
