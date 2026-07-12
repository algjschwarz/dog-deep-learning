import os
import random
import time
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import transformers
import csv
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer
load_dotenv()

def generate_labeled_instructions(amount=100) -> list:
    dog_scenarios = ["at the beach", "during a thunderstorm", "at the vet", "in the snow",
    "chasing a squirrel", "meeting a new puppy", "at the park", "getting a bath",
    "on a road trip", "waiting for dinner", "at the dog park", "learning a trick",
    "stealing food", "greeting its owner", "in the backyard", "sleeping",
    "digging a hole", "howling at the moon", "riding in a car", "playing tug of war",
    "rolling in mud", "catching a frisbee", "begging at the table", "napping on the couch",
    "chewing a shoe", "barking at the doorbell", "running on the trail", "swimming in a lake",
    "hiding from fireworks", "wearing a costume", "visiting the groomer", "playing with a toy",
    "guarding the house", "waiting by the window", "burying a bone", "shaking off water",
    "chasing its tail", "meeting another dog", "getting belly rubs", "on a hiking trip",
    "at the beach at sunset", "in a snowstorm", "at obedience school", "on the farm",
    "herding sheep", "fetching the newspaper", "licking an ice cream cone", "in a backpack carrier",
    "at a birthday party", "sniffing everything on a walk", "curled up by the fire", "playing in the sprinkler",
    "waiting outside a store", "riding on a boat", "meeting a baby", "in a thunderstorm hiding under the bed",
    "learning to roll over", "at the pet store", "watching squirrels through the window", "getting a new collar"]

    non_dog_topics = ["a rainstorm", "a city street", "cooking dinner", "a mountain hike",
        "a broken car", "a birthday party", "the stock market", "a library",
        "a spaceship", "an old photograph", "a cup of coffee", "a traffic jam",
        "a snowy morning", "a crowded subway", "a quiet forest", "an ocean wave",
        "a bakery", "a thunderstorm over the sea", "an autumn leaf", "a distant planet",
        "a busy airport", "a summer garden", "an antique clock", "a mountain river",
        "a candle flame", "a chess game", "a violin concert", "a sandcastle",
        "a lighthouse", "a farmers market", "a snow-capped peak", "an empty highway",
        "a bookstore", "a bowl of soup", "a rocket launch", "a foggy harbor",
        "a desert at night", "a jazz club", "a wheat field", "a broken umbrella",
        "a train station", "a cup of tea", "a full moon", "a waterfall",
        "a chess tournament", "a bustling cafe", "an old bridge", "a field of sunflowers",
        "a winter cabin", "a city at night", "a paper airplane", "a grand piano",
        "a coral reef", "a dusty attic", "a mountain sunrise", "a subway platform",
        "a stormy sky", "a quiet lake", "an old typewriter", "a busy kitchen"]
    labeled_instructions = []
    for i in range(amount):
        if i < amount * (1/4):
            scene = random.choice(dog_scenarios)
            labeled_instructions.append((f"Write one sentence about a dog {scene}. Only the sentence.", True))
        elif i < amount * (1/2):
            scene = random.choice(dog_scenarios)
            labeled_instructions.append((f"Write one sentence about a dog {scene} without using the word dog. Only the sentence.", True))
        else:
            topic = random.choice(non_dog_topics)
            labeled_instructions.append((f"Write one sentence about {topic}. Only the sentence.", False))
    return labeled_instructions

def create_csv_file(data):
    file_path = 'labeled_data.csv'
    with open(file_path, mode='a', newline='') as file:
        csv.writer(file).writerows(data)

def generate_text_using_api(instruction, seed):
    model = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    data = model.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{instruction}"
            }
        ],
        model="llama-3.1-8b-instant",
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
            generated_text = generate_text_using_api(instruction, index)
            time.sleep(2.1)
        labeled_data.append((generated_text, label))
        print(f"{generated_text} seed {index}")
        index += 1
    create_csv_file(labeled_data)
    print("5. done, csv written")

def setup_vectorizer() -> CountVectorizer:
    vectorizer = CountVectorizer(analyzer='word')
    data = []
    with open('labeled_data.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row[0])
    if len(data) == 0:
        raise RuntimeError("Data not read")
    vectorizer.fit_transform(data)
    return vectorizer

def get_data() -> list:
    data = []
    with open('labeled_data.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data

class Dog_or_Not(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()

        self.input = nn.Linear(vocabulary_size, 20)
        self.output = nn.Linear(20, 1)

    def forward(self, input):
        l1 = F.relu(self.input(input))
        l2 = self.output(l1)
        return l2

def train_model(vectorizer, epochs, training_data, net, loss_function, optimizer):
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for (text, label) in training_data:
            tokenized_text = torch.tensor(vectorizer.transform([text]).toarray(), dtype=torch.float32)
            out = net(tokenized_text)
            loss = loss_function(out, torch.tensor([[1.0 if label == "True" else 0.0]]))
            loss.backward()
            optimizer.step()
            net.zero_grad()
        random.shuffle(training_data)

def test_model(test_data, vectorizer, net):
    correct = 0
    for (text, label) in test_data:
        tokenized_text = torch.tensor(vectorizer.transform([text]).toarray(), dtype=torch.float32)
        out = net(tokenized_text)
        prob = torch.sigmoid(out).item()
        result = prob >= 0.50
        if result == (label == "True"):
            correct += 1
    print(f"Accuracy: {(correct / len(test_data)) * 100}%")

def main():
    create_synthetic_data(amount=500, use_api=True)
    epochs = 10
    learning_rate=0.01
    vectorizer = setup_vectorizer()
    net = Dog_or_Not(len(vectorizer.vocabulary_))
    data = get_data()
    random.shuffle(data)
    training_data = data[:len(data)//2]
    test_data = data[len(data)//2:]
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    train_model(vectorizer, epochs, training_data, net, loss_function, optimizer)
    test_model(test_data, vectorizer, net)
    

if __name__ == "__main__":
    main()
