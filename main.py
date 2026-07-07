import transformers
import torch

def generate_synthetic_data():
    if torch.cuda.is_available():
        pipe = transformers.pipeline("text-generation", "Qwen/Qwen3-0.6B", device="cuda")
        print(pipe("hello"))
    else:
        raise RuntimeError("No gpu detected")

def main():
    generate_synthetic_data()

main()
