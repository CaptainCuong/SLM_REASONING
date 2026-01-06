"""
Compute gradient projections for JSON datasets in the data folder.
This script processes all examples from JSON files and saves projected gradients.
"""

import torch
import torch.nn as nn
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add SML_REASONING directory to path
sys.path.append(os.getcwd())

from ghostEngines.gradProjection.gradproj_engine import GradProjLoraEngine


def load_model(model_name):
    """Load model and tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("transformers library required. Install with: pip install transformers")

    print(f"Loading {model_name} model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_json_dataset(json_file_path):
    """Load dataset from a JSON file in the data folder."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    print(f"Loading dataset from {json_file_path}...")

    # Check if file exists
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File not found: {json_file_path}")

    dataset = load_dataset("json", data_files=json_file_path, split='train')

    print(f"Dataset loaded. Total examples: {len(dataset)}")
    return dataset


def prepare_prompts(data, tokenizer):
    """
    Prepare prompts from data entries.

    Args:
        data: List of entries with 'instruction' and 'input' fields
        tokenizer: The tokenizer to use

    Returns:
        List of formatted prompts
    """
    prompts = []

    for entry in data:
        instruction = entry.get("instruction", "")
        question = entry.get("input", "")

        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": f"{instruction}\n\n{question}"}
        ]

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    return prompts


def prepare_batch_from_dataset(examples, tokenizer, max_length=512):
    """
    Prepare a batch of examples from the dataset.
    Only calculate loss on the 'output' field.

    Args:
        examples: List of dataset examples (with 'instruction', 'input', 'output' fields)
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        inputs: Tokenized inputs
        labels: Labels for language modeling (loss only on output)
    """
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    # Use prepare_prompts to format inputs
    prompts = prepare_prompts(examples, tokenizer)

    for idx, example in enumerate(examples):
        # Get formatted prompt from prepare_prompts
        input_text = prompts[idx]
        output_text = example['output']

        # Tokenize input and output separately
        input_ids = tokenizer.encode(input_text, add_special_tokens=True)
        output_ids = tokenizer.encode(output_text, add_special_tokens=False)

        # Combine input + output
        combined_ids = input_ids + output_ids

        # Truncate if necessary
        if len(combined_ids) > max_length:
            # Keep full input if possible, truncate output
            if len(input_ids) < max_length:
                output_ids = output_ids[:max_length - len(input_ids)]
                combined_ids = input_ids + output_ids
            else:
                # Input alone exceeds max_length, truncate everything
                combined_ids = combined_ids[:max_length]
                output_ids = []  # All output was truncated

        # Create labels: -100 for input (no loss), actual tokens for output
        # Labels should match the actual combined_ids used
        input_len = len(combined_ids) - len(output_ids)
        labels = [-100] * input_len + output_ids
        assert len(labels) == len(combined_ids), f"Labels length {len(labels)} != combined_ids length {len(combined_ids)}"

        batch_input_ids.append(combined_ids)
        batch_labels.append(labels)

    # Pad sequences
    max_len = max(len(ids) for ids in batch_input_ids)

    for i in range(len(batch_input_ids)):
        padding_length = max_len - len(batch_input_ids[i])

        # Pad input_ids
        batch_input_ids[i] = batch_input_ids[i] + [tokenizer.pad_token_id] * padding_length

        # Pad labels with -100
        batch_labels[i] = batch_labels[i] + [-100] * padding_length

        # Create attention mask
        attention_mask = [1] * (max_len - padding_length) + [0] * padding_length
        batch_attention_mask.append(attention_mask)

    # Convert to tensors
    inputs = {
        'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long)
    }
    
    labels = torch.tensor(batch_labels, dtype=torch.long)

    return inputs, labels

def process_batch(model, engine, tokenizer, batch_examples, batch_idx, args, device):
    """Process a single batch and compute gradient projections."""
    try:
        # Print GPU memory before processing
        if device.type == 'cuda':
            torch.cuda.synchronize()
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
            print(f"\nBatch {batch_idx} - GPU Memory Before: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        # Prepare batch
        inputs, labels = prepare_batch_from_dataset(
            batch_examples, tokenizer, max_length=args.max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        # Zero gradients
        model.zero_grad()

        # Forward pass
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Collect projected gradients
        # Adjust indices based on start_index to ensure correct global indices
        start_offset = args.start_index if hasattr(args, 'start_index') else 0
        batch_indices = list(range(
            start_offset + batch_idx * args.batch_size,
            start_offset + batch_idx * args.batch_size + len(batch_examples)
        ))
        projections = engine.collect_batch(batch_indices)

        # Clear cached gradients and activations to prevent memory accumulation
        engine.clear_gradients()

        # Explicitly clean up tensors
        del outputs, loss, inputs, labels, projections
        model.zero_grad()

        # Clear CUDA cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n⚠ Error processing batch {batch_idx}: {e}")
        print("Skipping this batch...")
        raise e
    
def run_gradproj_on_dataset(args):
    """Run gradient projection on JSON dataset."""
    print("\n=== JSON Dataset Gradient Projection ===")
    print(f"Processing file: {args.json_file}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, tokenizer = load_model(args.model_name)
    model = model.to(device)
    model.eval()  # Use eval mode to avoid dropout

    # Disable caching for gradient computation
    model.config.use_cache = False

    # Apply transformers support if available
    try:
        from ghostEngines import transformers_support
        transformers_support.forward_swapper(model)
        print("Applied transformers forward swapper")
    except ImportError:
        print("Warning: transformers_support not available, continuing without it")
    
    # Load dataset
    dataset = load_json_dataset(args.json_file)

    # Apply start/end index slicing
    dataset_length = len(dataset)
    start_idx = max(0, args.start_index)
    end_idx = min(dataset_length, args.end_index) if args.end_index > 0 else dataset_length

    if start_idx >= dataset_length:
        raise ValueError(f"start_index {start_idx} is beyond dataset length {dataset_length}")

    dataset = dataset.select(range(start_idx, end_idx))
    print(f"Processing dataset slice: [{start_idx}:{end_idx}] ({len(dataset)} examples)")

    # Calculate total number of batches
    total_examples = len(dataset)
    if args.max_examples > 0:
        total_examples = min(total_examples, args.max_examples)
    total_batches = (total_examples + args.batch_size - 1) // args.batch_size

    print(f"Batch size: {args.batch_size}")
    print(f"Total batches to process: {total_batches}")

    # Update proj_dir to include file name and index range
    json_file_name = Path(args.json_file).stem  # Get filename without extension
    proj_dir_with_range = f"{args.proj_dir}_{json_file_name}_idx{args.start_index}_{args.end_index if args.end_index > 0 else 'end'}"
    print(f"Save directory: {proj_dir_with_range}")

    # Create projection engine
    engine_config = {
        'proj_layers': args.proj_layers,
        'proj_rank_total': args.proj_rank_total,
        'proj_rank_min': args.proj_rank_min,
        'proj_seed': args.proj_seed,
        'proj_dtype': args.proj_dtype,
        'proj_dir': proj_dir_with_range,
        'proj_save_interval': args.proj_save_interval,
        'include_embeddings': args.include_embeddings,
    }

    print("\nInitializing projection engine...")
    engine = GradProjLoraEngine(model, **engine_config)
    engine.attach()

    # Process dataset in batches
    batch_examples = []
    batch_idx = 0
    total_processed = 0

    print("\nProcessing dataset...")
    dataset_iter = iter(dataset)

    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        while True:
            # Collect batch
            try:
                example = next(dataset_iter)
                batch_examples.append(example)
                total_processed += 1

                # Check if we've reached max examples
                if args.max_examples > 0 and total_processed >= args.max_examples:
                    if batch_examples:  # Process remaining examples
                        process_batch(
                            model, engine, tokenizer, batch_examples,
                            batch_idx, args, device
                        )
                        pbar.update(1)
                    break

            except StopIteration:
                # End of dataset
                if batch_examples:  # Process remaining examples
                    process_batch(
                        model, engine, tokenizer, batch_examples,
                        batch_idx, args, device
                    )
                    pbar.update(1)
                break

            # Process batch when full
            if len(batch_examples) >= args.batch_size:
                try:
                    process_batch(
                        model, engine, tokenizer, batch_examples,
                        batch_idx, args, device
                    )
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    raise e
                
                batch_examples = []
                batch_idx += 1
                pbar.update(1)

    # Detach engine
    engine.detach()

    print(f"\n✓ Complete! Processed {total_processed} examples in {batch_idx + 1} batches")
    print(f"✓ Projections saved to: {proj_dir_with_range}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute gradient projections for JSON datasets in data folder'
    )

    # Model configuration
    parser.add_argument(
        '--model_name',
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help='Model name or path'
    )

    # Dataset configuration
    parser.add_argument(
        '--json_file',
        type=str,
        required=True,
        help='Path to JSON file in data folder (e.g., data/algebra.json)'
    )
    parser.add_argument(
        '--max_examples',
        type=int,
        default=-1,
        help='Maximum number of examples to process (-1 for all)'
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='Starting index of the dataset to process (default: 0)'
    )
    parser.add_argument(
        '--end_index',
        type=int,
        default=-1,
        help='Ending index of the dataset to process (-1 for all, default: -1)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=2000,
        help='Maximum sequence length'
    )

    # Projection configuration (required)
    parser.add_argument(
        '--proj_layers',
        type=str,
        default='self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,mlp.up_proj,mlp.gate_proj',
        help='Comma-separated layer name patterns'
    )
    parser.add_argument(
        '--proj_rank_total',
        type=int,
        default=512,
        help='Target total projection dimension per layer'
    )
    parser.add_argument(
        '--proj_rank_min',
        type=int,
        default=16,
        help='Minimum dimension for k_i and k_o'
    )
    parser.add_argument(
        '--proj_seed',
        type=int,
        default=42,
        help='Random seed for projection matrices'
    )
    parser.add_argument(
        '--proj_dtype',
        type=str,
        default='bfloat16',
        choices=['float16', 'bfloat16', 'float32'],
        help='Data type for storing projections'
    )
    parser.add_argument(
        '--proj_dir',
        type=str,
        default='/projects/ai_safe/cuongdc/grad_proj',
        help='Directory to save projections'
    )
    parser.add_argument(
        '--proj_save_interval',
        type=int,
        default=1,
        help='Save projections every N iterations'
    )
    parser.add_argument(
        '--include_embeddings',
        action='store_true',
        help='Include embedding layers in projection'
    )

    # Data configuration
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size for processing'
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.proj_layers:
        raise ValueError("--proj_layers is required")
    if args.proj_rank_total <= 0:
        raise ValueError("--proj_rank_total must be positive")
    if args.proj_rank_min <= 0:
        raise ValueError("--proj_rank_min must be positive")

    # Create output directory
    Path(args.proj_dir).mkdir(parents=True, exist_ok=True)

    # Run projection
    run_gradproj_on_dataset(args)


if __name__ == '__main__':
    main()