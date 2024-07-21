import json

with open("training.txt", mode="r", encoding="utf-8") as f:
    content = f.readlines()

intent = "no_match"

training = []

for line in content:
    training.append(
        {
            "count": 1,
            "intent": intent,
            "patterns": [
                {"text": line.strip().lower().replace(".", "").replace("?", "")}
            ],
        }
    )

all_training = {"utterances": training, "entities": []}

with open(f"model_training/{intent}.json", mode="w+", encoding="utf-8") as f:
    json.dump(all_training, f, indent=4)
