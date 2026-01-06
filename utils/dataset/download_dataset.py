#!/usr/bin/env python3
"""
Script to download math datasets from Hugging Face.
"""

from datasets import load_dataset
import os

data_name = "allenai/tulu-3-sft-mixture"
# ["bespokelabs/Bespoke-Stratos-17k", "openai/gsm8k_main",
# "openai/gsm8k_socratic", "Hothan/OlympiadBench",
# "AI-MO/NuminaMath-CoT", "HuggingFaceH4/MATH-500",
# "open-thoughts/OpenThoughts-114k", "open-r1/OpenR1-Math-220k",
# "allenai/tulu-3-sft-mixture", "tatsu-lab/alpaca_eval",
# "newfacade/LeetCodeDataset", "openai/openai_humaneval",
# "Muennighoff/mbpp", "TIGER-Lab/TheoremQA"]

if data_name == "bespokelabs/Bespoke-Stratos-17k":
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k")
elif data_name == "openai/gsm8k_main":
    dataset = load_dataset("openai/gsm8k", "main")
elif data_name == "openai/gsm8k_socratic":
    dataset = load_dataset("openai/gsm8k", "socratic")
elif data_name == "Hothan/OlympiadBench":
    dataset = load_dataset("Hothan/OlympiadBench")
elif data_name == "AI-MO/NuminaMath-CoT":
    dataset = load_dataset("AI-MO/NuminaMath-CoT")
elif data_name == "HuggingFaceH4/MATH-500":
    dataset = load_dataset("HuggingFaceH4/MATH-500")
elif data_name == "open-thoughts/OpenThoughts-114k":
    dataset = load_dataset("open-thoughts/OpenThoughts-114k")
elif data_name == "open-r1/OpenR1-Math-220k":
    dataset = load_dataset("open-r1/OpenR1-Math-220k")
elif data_name == "allenai/tulu-3-sft-mixture":
    dataset = load_dataset("allenai/tulu-3-sft-mixture")
elif data_name == "tatsu-lab/alpaca_eval":
    dataset = load_dataset("tatsu-lab/alpaca_eval")
elif data_name == "newfacade/LeetCodeDataset":
    dataset = load_dataset("newfacade/LeetCodeDataset")
elif data_name == "openai/openai_humaneval":
    dataset = load_dataset("openai/openai_humaneval")
elif data_name == "Muennighoff/mbpp":
    dataset = load_dataset("Muennighoff/mbpp")
elif data_name == "TIGER-Lab/TheoremQA":
    dataset = load_dataset("TIGER-Lab/TheoremQA")
else:
    raise ValueError(f"Unknown dataset name: {data_name}")

print("\nDataset downloaded successfully!")
print(f"\nDataset structure:")
print(dataset)

print("\nDataset downloaded successfully!")
print(f"\nDataset structure:")
print(dataset)
