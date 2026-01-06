#!/usr/bin/env python3
"""
Script to calculate perplexity of an LLM on given sentences.
Perplexity measures how well a language model predicts text - lower is better.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datasets import load_dataset
from utils.generate_answer import generate_answer_transformers

def calculate_perplexity(model, tokenizer, text, device="cuda"):
    """
    Calculate perplexity for a given text using a language model.

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text string
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        perplexity: The perplexity score
        loss: The cross-entropy loss
    """
    # Tokenize the text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    # Get model output
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    # Calculate perplexity: exp(loss)
    perplexity = np.exp(loss)

    return perplexity, loss

dataset = load_dataset("open-r1/OpenR1-Math-220k")

sentence_1 = dataset['train'][0]['solution']
sentence_2 = dataset['train'][0]['generations'][0]
sentence_3 = generate_answer_transformers(dataset['train'][0]['problem'], model_name="/projects/ai_safe/cuongdc/open_r1/algebra/checkpoint-375/", top_p=1.0)
sentence_4 = generate_answer_transformers(dataset['train'][0]['problem'], model_name="/projects/ai_safe/cuongdc/open_r1/algebra_generated/checkpoint-375/", top_p=1.0)
sentence_5 = generate_answer_transformers(dataset['train'][0]['problem'], model_name="/projects/ai_safe/cuongdc/open_r1/algebra_generated/checkpoint-1625/", top_p=1.0)
sentence_6 = generate_answer_transformers(dataset['train'][0]['problem'], model_name="Qwen/Qwen2.5-Math-7B", top_p=1.0)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-7B",
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically distribute model across available GPUs
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Math-7B",
    trust_remote_code=True
)
pplx_1, loss_1 = calculate_perplexity(model, tokenizer, sentence_1)
pplx_2, loss_2 = calculate_perplexity(model, tokenizer, sentence_2)
pplx_3, loss_3 = calculate_perplexity(model, tokenizer, sentence_3)
pplx_4, loss_4 = calculate_perplexity(model, tokenizer, sentence_4)
pplx_5, loss_5 = calculate_perplexity(model, tokenizer, sentence_5)
pplx_6, loss_6 = calculate_perplexity(model, tokenizer, sentence_6)

print(f"Perplexity 1: {pplx_1:.2f}, Loss 1: {loss_1:.4f}\n")
print(f"Perplexity 2: {pplx_2:.2f}, Loss 2: {loss_2:.4f}\n")
print(f"Perplexity 3: {pplx_3:.2f}, Loss 3: {loss_3:.4f}\n")
print(f"Perplexity 4: {pplx_4:.2f}, Loss 4: {loss_4:.4f}\n")
print(f"Perplexity 5: {pplx_5:.2f}, Loss 5: {loss_5:.4f}\n")
print(f"Perplexity 6: {pplx_6:.2f}, Loss 6: {loss_6:.4f}\n")