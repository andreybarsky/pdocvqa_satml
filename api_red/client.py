import os
import time
import json
import requests
import argparse

class QueryClient:
    def __init__(self, url, user_token, data_path, resp_saving_path):
        self.url = url
        self.user_token = user_token
        self.data_path = data_path
        self.resp_saving_path = resp_saving_path
        self.headers = {
            'Content-Type': 'application/json',
            'Bearer': self.user_token
        }

    def load_data(self):
        try:
            with open(self.data_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {self.data_path}")

    def save_response(self, response, query_id):
        os.makedirs(self.resp_saving_path, exist_ok=True)
        response_path = os.path.join(self.resp_saving_path, f"{query_id}.json")
        with open(response_path, "w") as f:
            json.dump(response, f, indent=4)
        return response_path

    def post_query(self, data):
        try:
            response = requests.post(self.url, json=data, headers=self.headers)
            return response
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to communicate with the server: {e}")

    def process_response(self, response, query_id):
        if response.status_code == 200:
            print("Token valid. Remaining credits for the hour:", response.json().get("remaining_credit"))
            response_path = self.save_response(response.json(), query_id)
            print(f"Results saved in {response_path}")
        elif response.status_code == 401:
            print("Token not valid")
        elif response.status_code == 403:
            print("Submitted JSON format not valid:", response.json())
        elif response.status_code == 405:
            print("Submitted JSON not valid")
        elif response.status_code == 402:
            print("Communication error")
        else:
            print(f"Undefined error. Status code: {response.status_code}, Response: {response.text}")

    def run(self):
        data = self.load_data()
        query_id = str(int(time.time() * 1_000_000))
        response = self.post_query(data)
        self.process_response(response, query_id)

if __name__ == "__main__":
    
    URL = "http://158.109.8.119:8197/query"

    parser = argparse.ArgumentParser(description="QueryClient to send JSON queries to a server.")
    parser.add_argument("--token", type=str, required=True, help="User authentication token.")
    parser.add_argument("--query_path", type=str, required=True, help="Path to the JSON query file.")
    parser.add_argument("--response_save_path", type=str, default="./query_result/", help="Path to save the response JSON files.")

    args = parser.parse_args()

    client = QueryClient(
        url=URL,
        user_token=args.token,
        data_path=args.query_path,
        resp_saving_path=args.response_save_path
    )
    client.run()