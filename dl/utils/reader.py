import yaml

def load_yaml(filepath: str):
    with open(filepath, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config