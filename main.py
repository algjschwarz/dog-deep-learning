import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import csv

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

def create_synthetic_data(amount=100):
    print("1. starting")
    if not torch.cuda.is_available():
        raise RuntimeError("No gpu detected")
    print("2. gpu check passed")
    pipe = transformers.pipeline("text-generation", "Qwen/Qwen3-4B-Instruct-2507", device="cuda")
    print("3. model loaded")
    labeled_instructions = generate_labeled_instructions(amount=amount)
    print("4. instructions built, starting loop")
    labeled_data = []
    index = 0
    for instruction, label in labeled_instructions:
        generated_text = pipe(instruction, return_full_text=False, max_new_tokens=30)[0]['generated_text'].strip()
        labeled_data.append((generated_text, label))
        print(f"{generated_text} trial {index}")
        index += 1
    create_csv_file(labeled_data)
    print("5. done, csv written")

class Dog_or_Not(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        pass

def main():
    create_synthetic_data()

main()
