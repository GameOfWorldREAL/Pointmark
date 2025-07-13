import json
from pathlib import Path

from src.utils.utils import path_to_str

class JsonManager:
    """
    A class to manage JSON data storage and retrieval.

    The JsonManager class provides functionality to manage JSON files, including saving,
    loading, updating, and managing key-value pairs. It is designed primarily for easy
    interaction with JSON files. It initializes a JSON file based on the provided path
    and filename and offers methods to manipulate and retrieve its data.

    :ivar filename: The name of the JSON file.
    :type filename: str

    :ivar filepath: The full path to the JSON file, combining the provided directory path
        and filename.
    :type filepath: Path
    """
    def __init__(self,  path: Path, filename: str):
        self.filename = filename
        self.filepath = path / filename
        if not self.filepath.exists():
            self._save({})

    def __str__(self):
        data = self.load()
        data_str = ""
        for key, value in data.items():
            data_str+=f"{key}: {value}\n"
        return data_str[:-1]

    def __repr__(self):
        return str(self.load())

    def _save(self, data):
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self) -> dict:
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(path_to_str(self.filepath) + " does not exist")

    def update(self, update_data: dict):
         data = self.load()
         data.update(update_data)
         self._save(data)

    def set_entry(self, key, value):
        data = self.load()
        data[key] = value
        self._save(data)

    def get_value(self, key):
        data = self.load()
        if (not key is None) and (key in data):
            return data[key]
        return None

    def is_empty(self) -> bool:
        data = self.load()
        if data == {}:
            return True
        else:
            for key in data:
                if data[key] is not None:
                    return False
            return True




