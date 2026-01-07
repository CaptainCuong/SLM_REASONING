"""
Metrics calculation for language model responses.

This module provides functions to calculate various probability-based metrics
for evaluating language model responses, including perplexity, likelihood,
entropy, and token-level probabilities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math


@dataclass
class ResponseMetrics:
    """Container for response-level metrics."""
    perplexity: float
    log_likelihood: float
    average_token_prob: float
    entropy: float
    token_count: int
    token_probs: List[float]
    token_log_probs: List[float]
    token_entropies: List[float]
    tokens: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'perplexity': self.perplexity,
            'log_likelihood': self.log_likelihood,
            'average_token_prob': self.average_token_prob,
            'entropy': self.entropy,
            'token_count': self.token_count,
            'token_probs': self.token_probs,
            'token_log_probs': self.token_log_probs,
            'token_entropies': self.token_entropies,
            'tokens': self.tokens
        }


def calculate_token_probabilities(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    return_all_probs: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Calculate probabilities for given tokens from logits.

    Args:
        logits: Logits tensor of shape (batch_size, seq_len, vocab_size) or (seq_len, vocab_size)
        token_ids: Token IDs tensor of shape (batch_size, seq_len) or (seq_len,)
        return_all_probs: If True, return all token probabilities, not just selected ones

    Returns:
        Token probabilities (and optionally all probabilities)
    """
    # Ensure logits are 3D: (batch, seq_len, vocab_size)
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)

    # Calculate probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Gather probabilities for the actual tokens
    batch_size, seq_len = token_ids.shape
    token_probs = probs.gather(
        dim=-1,
        index=token_ids.unsqueeze(-1)
    ).squeeze(-1)

    if return_all_probs:
        return token_probs, probs
    return token_probs


def calculate_log_likelihood(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    reduction: str = 'mean'
) -> float:
    """
    Calculate log-likelihood of a sequence.

    Args:
        logits: Logits tensor of shape (batch_size, seq_len, vocab_size) or (seq_len, vocab_size)
        token_ids: Token IDs tensor of shape (batch_size, seq_len) or (seq_len,)
        reduction: How to reduce the log-likelihoods ('mean', 'sum', or 'none')

    Returns:
        Log-likelihood value(s)
    """
    # Get token probabilities
    token_probs = calculate_token_probabilities(logits, token_ids)

    # Calculate log probabilities
    log_probs = torch.log(token_probs + 1e-10)  # Add small epsilon to avoid log(0)

    if reduction == 'mean':
        return log_probs.mean().item()
    elif reduction == 'sum':
        return log_probs.sum().item()
    elif reduction == 'none':
        return log_probs
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def calculate_perplexity(
    logits: torch.Tensor,
    token_ids: torch.Tensor
) -> float:
    """
    Calculate perplexity of a sequence.

    Perplexity is exp(-average log-likelihood). Lower perplexity indicates
    the model is more confident about the sequence.

    Args:
        logits: Logits tensor of shape (batch_size, seq_len, vocab_size) or (seq_len, vocab_size)
        token_ids: Token IDs tensor of shape (batch_size, seq_len) or (seq_len,)

    Returns:
        Perplexity value
    """
    avg_log_likelihood = calculate_log_likelihood(logits, token_ids, reduction='mean')
    perplexity = math.exp(-avg_log_likelihood)
    return perplexity


def calculate_entropy(
    logits: torch.Tensor,
    reduction: str = 'mean'
) -> Union[float, torch.Tensor]:
    """
    Calculate entropy of probability distributions.

    Entropy measures uncertainty in the model's predictions.
    Higher entropy means more uncertainty.

    Args:
        logits: Logits tensor of shape (batch_size, seq_len, vocab_size) or (seq_len, vocab_size)
        reduction: How to reduce the entropies ('mean', 'sum', or 'none')

    Returns:
        Entropy value(s)
    """
    # Ensure logits are 3D
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)

    # Calculate probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Calculate entropy: -sum(p * log(p))
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1)

    if reduction == 'mean':
        return entropy.mean().item()
    elif reduction == 'sum':
        return entropy.sum().item()
    elif reduction == 'none':
        return entropy
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def calculate_response_metrics(
    model,
    tokenizer,
    instruction: str,
    response: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> ResponseMetrics:
    """
    Calculate comprehensive metrics for a model response.

    Args:
        model: The language model
        tokenizer: The tokenizer
        instruction: The instruction/prompt text
        response: The response text to evaluate
        device: Device to run calculations on

    Returns:
        ResponseMetrics object containing all calculated metrics
    """
    # Tokenize instruction and response
    instruction_ids = tokenizer.encode(instruction, add_special_tokens=True, return_tensors='pt')
    full_text = instruction + response
    full_ids = tokenizer.encode(full_text, add_special_tokens=True, return_tensors='pt')

    # Move to device
    full_ids = full_ids.to(device)

    # Get response token IDs (exclude instruction tokens)
    response_start_idx = instruction_ids.shape[1]
    response_ids = full_ids[:, response_start_idx:]

    # Get logits from model
    model.eval()
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    # Extract logits for response tokens (shift by 1 for next token prediction)
    response_logits = logits[:, response_start_idx-1:-1, :]

    # Calculate token probabilities
    token_probs = calculate_token_probabilities(response_logits, response_ids)
    token_log_probs = torch.log(token_probs + 1e-10)

    # Calculate entropies
    token_entropies = calculate_entropy(response_logits, reduction='none')

    # Calculate aggregate metrics
    log_likelihood = token_log_probs.sum().item()
    avg_token_prob = token_probs.mean().item()
    perplexity = math.exp(-token_log_probs.mean().item())
    entropy = token_entropies.mean().item()

    # Decode tokens for interpretability
    response_tokens = [
        tokenizer.decode([token_id])
        for token_id in response_ids.squeeze().tolist()
    ]

    return ResponseMetrics(
        perplexity=perplexity,
        log_likelihood=log_likelihood,
        average_token_prob=avg_token_prob,
        entropy=entropy,
        token_count=response_ids.shape[1],
        token_probs=token_probs.squeeze().tolist(),
        token_log_probs=token_log_probs.squeeze().tolist(),
        token_entropies=token_entropies.squeeze().tolist(),
        tokens=response_tokens
    )


def batch_calculate_metrics(
    model,
    tokenizer,
    instruction_response_pairs: List[Tuple[str, str]],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 8
) -> List[ResponseMetrics]:
    """
    Calculate metrics for multiple instruction-response pairs in batches.

    Args:
        model: The language model
        tokenizer: The tokenizer
        instruction_response_pairs: List of (instruction, response) tuples
        device: Device to run calculations on
        batch_size: Number of samples to process at once

    Returns:
        List of ResponseMetrics objects
    """
    all_metrics = []

    for i in range(0, len(instruction_response_pairs), batch_size):
        batch = instruction_response_pairs[i:i + batch_size]

        for instruction, response in batch:
            metrics = calculate_response_metrics(
                model, tokenizer, instruction, response, device
            )
            all_metrics.append(metrics)

    return all_metrics


def compare_responses(
    model,
    tokenizer,
    instruction: str,
    responses: List[str],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, ResponseMetrics]:
    """
    Compare multiple responses to the same instruction.

    Args:
        model: The language model
        tokenizer: The tokenizer
        instruction: The instruction/prompt text
        responses: List of response texts to compare
        device: Device to run calculations on

    Returns:
        Dictionary mapping response indices to their metrics
    """
    results = {}

    for idx, response in enumerate(responses):
        metrics = calculate_response_metrics(
            model, tokenizer, instruction, response, device
        )
        results[f"response_{idx}"] = metrics

    return results
