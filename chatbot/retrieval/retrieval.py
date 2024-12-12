import json


class Retrieval:
    def __init__(self, data_path: str = "./data/newjeans.json"):
        with open(data_path, "r") as file:
            data: dict[str, str] = json.load(file)
            self.data = data

    def retrieve(self, query: str) -> str | None:
        for key in self.data:
            if key.lower() in query.lower():
                return f"{key}: {self.data[key]}"
        return None

    def print_data(self) -> None:
        print(self.data)