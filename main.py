import os
import random
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import csv
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer
load_dotenv()

def generate_labeled_instructions(amount=100) -> list:
    dog_scenarios = ["at the beach", "during a thunderstorm", "at the vet", "in the snow",
        "chasing a squirrel", "meeting a new puppy", "at the park", "getting a bath",
        "on a road trip", "waiting for dinner", "at the dog park", "learning a trick",
        "stealing food", "greeting its owner", "in the backyard", "sleeping"]
    non_dog_topics = ["a rainstorm", "a city street", "cooking dinner", "a mountain hike",
        "a broken car", "a birthday party", "the stock market", "a library",
        "a spaceship", "an old photograph", "a cup of coffee", "a traffic jam"]
    labeled_instructions = []
    for i in range(amount):
        if i < amount * (1/3):
            scene = random.choice(dog_scenarios)
            labeled_instructions.append((f"Write one sentence about a dog {scene}. Only the sentence.", True))
        elif i < amount * (2/3):
            scene = random.choice(dog_scenarios)
            labeled_instructions.append((f"Write one sentence about a dog {scene} without using the words dog, puppy, or canine. Only the sentence.", True))
        else:
            topic = random.choice(non_dog_topics)
            labeled_instructions.append((f"Write one sentence about {topic}. Only the sentence.", False))
    return labeled_instructions

def create_csv_file(data):
    file_path = 'labeled_data.csv'
    with open(file_path, mode='a', newline='') as file:
        csv.writer(file).writerows(data)

def generate_text_api(instruction, seed):
    model = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    data = model.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{instruction}"
            }
        ],
        model="llama-3.3-70b-versatile",
        seed=seed,
        temperature=1.0
    )
    return data.choices[0].message.content

def create_synthetic_data(amount=100, use_api=False):
    print("starting")
    if not use_api:
        if not torch.cuda.is_available():
            raise RuntimeError("No gpu detected")
        print("gpu check passed")
        pipe = transformers.pipeline("text-generation", "Qwen/Qwen3-4B-Instruct-2507", device="cuda")
        print("model loaded")
    labeled_instructions = generate_labeled_instructions(amount=amount)
    print("instructions built, starting loop")
    labeled_data = []
    index = 0
    for instruction, label in labeled_instructions:
        if not use_api:
            generated_text = pipe(instruction, return_full_text=False, max_new_tokens=30)[0]['generated_text'].strip()
        else:
            generated_text = generate_text_api(instruction, index)
        labeled_data.append((generated_text, label))
        print(f"{generated_text} seed {index}")
        index += 1
    create_csv_file(labeled_data)
    print("5. done, csv written")

class Dog_or_Not(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()

        self.input = nn.Linear(vocabulary_size, 20)
        self.output = nn.Linear(20, 1)

    def forward(self, input):
        l1 = F.relu(self.input(input))
        l2 = self.output(l1)
        return l2

def main():
    #create_synthetic_data(use_api=True)
    vectorizer = CountVectorizer(analyzer='word')
    vectorizer.fit_transform(["goon", 'fortnite dih', "jeffery epstein"])
    net = Dog_or_Not(len(vectorizer.vocabulary_))
    input = torch.randn(5)
    output = net(input)
    



main()
