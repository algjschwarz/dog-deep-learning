import os
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
    labeled_instructions = []
    for i in range(amount):
        if i < amount * (1/3):
            labeled_instructions.append(("Give me a sentence about a dog. Only the sentence nothing else.", True))
        elif i < amount * (2/3):
            labeled_instructions.append(("give me a sentence about a dog that doesn't explicitly state that it is a dog. e.g. The puppy wouldn't stop barking all night," + 
            " He's a very good boy who fetches the newspaper every morning. Only the sentence nothing else", True))
        else:
            labeled_instructions.append(("give me a sentence. Only the sentence nothing else.", False))
    return labeled_instructions

def create_csv_file(data):
    file_path = 'labeled_data.csv'
    with open(file_path, mode='w', newline='') as file:
        csv.writer(file).writerows(data)

def generate_text_api(instruction):
    model = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    data = model.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{instruction}"
            }
        ],
        model="llama-3.3-70b-versatile"
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
            generated_text = generate_text_api(instruction)
        labeled_data.append((generated_text, label))
        print(f"{generated_text} trial {index}")
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
    create_synthetic_data(use_api=True)
    vectorizer = CountVectorizer(analyzer='word')
    vectorizer.fit_transform(["goon", 'fortnite dih', "jeffery epstein"])
    net = Dog_or_Not(len(vectorizer.vocabulary_))
    number = torch.randn(5)
    print(number)
    print(net.forward(number))

main()
