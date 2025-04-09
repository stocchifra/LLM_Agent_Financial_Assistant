import json

data_path = "/Users/francescostocchi/ConvFinQA_LLM_Project/data/train.json"

with open(data_path, "r") as f:
    data = json.load(f)

# print(data[0])

questions = [
    entry["qa"]["question"]
    for entry in data
    if "qa" in entry and "question" in entry["qa"]
]

print(questions[0])
