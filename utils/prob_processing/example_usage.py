"""
Example usage of probability processing metrics.

This script demonstrates how to use the metrics module to calculate
perplexity, likelihood, and other metrics for model responses.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from metrics import calculate_response_metrics, compare_responses, ResponseMetrics
import json


def example_single_response():
    """Example: Calculate metrics for a single response."""
    print("=" * 80)
    print("Example 1: Single Response Metrics")
    print("=" * 80)

    # Load model and tokenizer (use a small model for demo)
    model_name = "gpt2"  # Replace with your model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Example instruction and response
    instruction = "What is the capital of France?"
    response = " The capital of France is Paris."

    print(f"\nInstruction: {instruction}")
    print(f"Response: {response}")

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_response_metrics(model, tokenizer, instruction, response, device)

    # Display results
    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    print(f"Perplexity: {metrics.perplexity:.4f}")
    print(f"Log-Likelihood: {metrics.log_likelihood:.4f}")
    print(f"Average Token Probability: {metrics.average_token_prob:.4f}")
    print(f"Entropy: {metrics.entropy:.4f}")
    print(f"Token Count: {metrics.token_count}")

    print("\nPer-token metrics:")
    print("-" * 80)
    for i, (token, prob, log_prob, entropy) in enumerate(zip(
        metrics.tokens[:10],  # Show first 10 tokens
        metrics.token_probs[:10],
        metrics.token_log_probs[:10],
        metrics.token_entropies[:10]
    )):
        print(f"Token {i}: '{token}' | Prob: {prob:.4f} | Log-Prob: {log_prob:.4f} | Entropy: {entropy:.4f}")

    if len(metrics.tokens) > 10:
        print(f"... and {len(metrics.tokens) - 10} more tokens")


def example_compare_responses():
    """Example: Compare multiple responses to the same instruction."""
    print("\n\n" + "=" * 80)
    print("Example 2: Compare Multiple Responses")
    print("=" * 80)

    # Load model and tokenizer
    model_name = "gpt2"
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Example instruction with multiple responses
    instruction = "Explain photosynthesis:"
    responses = [
        " Photosynthesis is the process by which plants convert sunlight into energy.",
        " Plants use sunlight to make food through photosynthesis.",
        " The sun helps plants grow by photosynthesis which makes glucose."
    ]

    print(f"\nInstruction: {instruction}")
    print("\nResponses to compare:")
    for i, resp in enumerate(responses):
        print(f"  {i+1}.{resp}")

    # Compare responses
    print("\nCalculating metrics for all responses...")
    results = compare_responses(model, tokenizer, instruction, responses, device)

    # Display comparison
    print("\n" + "-" * 80)
    print("COMPARISON RESULTS:")
    print("-" * 80)
    print(f"{'Response':<12} {'Perplexity':<12} {'Log-Lik':<12} {'Avg Prob':<12} {'Entropy':<12}")
    print("-" * 80)

    for resp_id, metrics in results.items():
        print(f"{resp_id:<12} {metrics.perplexity:<12.4f} {metrics.log_likelihood:<12.4f} "
              f"{metrics.average_token_prob:<12.4f} {metrics.entropy:<12.4f}")

    # Find best response (lowest perplexity)
    best_resp_id = min(results.keys(), key=lambda k: results[k].perplexity)
    print(f"\nBest response (lowest perplexity): {best_resp_id}")


def example_save_metrics():
    """Example: Save metrics to JSON file."""
    print("\n\n" + "=" * 80)
    print("Example 3: Save Metrics to File")
    print("=" * 80)

    # Load model and tokenizer
    model_name = "gpt2"
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Example data
    instruction = "What is 2 + 2?"
    response = " 2 + 2 equals 4."

    print(f"\nInstruction: {instruction}")
    print(f"Response: {response}")

    # Calculate metrics
    metrics = calculate_response_metrics(model, tokenizer, instruction, response, device)

    # Save to JSON
    output_file = "/workspace/SLM_REASONING/utils/prob_processing/example_metrics.json"
    metrics_dict = metrics.to_dict()

    with open(output_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\nMetrics saved to: {output_file}")
    print("\nSample of saved data:")
    print(json.dumps({k: v for k, v in list(metrics_dict.items())[:5]}, indent=2))


def example_batch_processing():
    """Example: Process multiple instruction-response pairs."""
    print("\n\n" + "=" * 80)
    print("Example 4: Batch Processing")
    print("=" * 80)

    # Load model and tokenizer
    model_name = "gpt2"
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Example dataset
    pairs = [
        ("What is the capital of France?", " Paris is the capital of France."),
        ("What is 5 x 6?", " 5 x 6 equals 30."),
        ("Who wrote Hamlet?", " William Shakespeare wrote Hamlet."),
    ]

    print(f"\nProcessing {len(pairs)} instruction-response pairs...")

    # Process all pairs
    from metrics import batch_calculate_metrics
    all_metrics = batch_calculate_metrics(model, tokenizer, pairs, device, batch_size=2)

    # Display summary
    print("\n" + "-" * 80)
    print("BATCH PROCESSING RESULTS:")
    print("-" * 80)
    print(f"{'Pair':<6} {'Perplexity':<12} {'Log-Lik':<12} {'Tokens':<8}")
    print("-" * 80)

    for i, metrics in enumerate(all_metrics):
        print(f"{i+1:<6} {metrics.perplexity:<12.4f} {metrics.log_likelihood:<12.4f} {metrics.token_count:<8}")

    avg_perplexity = sum(m.perplexity for m in all_metrics) / len(all_metrics)
    print(f"\nAverage Perplexity: {avg_perplexity:.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Probability Processing Metrics - Examples")
    print("=" * 80)
    print("\nNote: These examples use GPT-2 for demonstration.")
    print("Replace with your own model for actual use.\n")

    # Run examples
    example_single_response()
    example_compare_responses()
    example_save_metrics()
    example_batch_processing()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
