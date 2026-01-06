"""
List all unique keywords in the 'source' column from the dataset.
Choose randomly 1 example from each category and save in JSON format.
"""

from datasets import load_dataset
from collections import Counter, defaultdict
import random
import json

# Dataset to analyze
data_name = "allenai/tulu-3-sft-mixture"

print(f"Loading dataset: {data_name}")
print("This may take a while for large datasets...")

# Load the dataset
dataset = load_dataset(data_name)

print(f"\nDataset loaded successfully!")
print(f"Dataset structure: {dataset}")

# Check which splits are available
print(f"\nAvailable splits: {list(dataset.keys())}")

# Analyze the source column from all splits
all_sources = []
all_samples = []  # Store all samples for random selection

for split_name, split_data in dataset.items():
    print(f"\nAnalyzing split: {split_name}")
    print(f"Number of samples: {len(split_data)}")

    # Check if 'source' column exists
    if 'source' in split_data.column_names:
        sources = split_data['source']
        all_sources.extend(sources)
        # Store samples with their source
        for sample in split_data:
            all_samples.append(sample)
        print(f"Found {len(sources)} source values")
    else:
        print(f"Available columns: {split_data.column_names}")
        print("WARNING: 'source' column not found in this split!")

if all_sources:
    # Count unique sources
    source_counter = Counter(all_sources)
    unique_sources = sorted(source_counter.keys())

    print("\n" + "="*60)
    print("UNIQUE SOURCE KEYWORDS")
    print("="*60)

    print(f"\nTotal unique sources: {len(unique_sources)}")
    print("\nList of all unique source keywords:")
    for i, source in enumerate(unique_sources, 1):
        count = source_counter[source]
        print(f"{i:3d}. {source:50s} (count: {count:,})")

    print("\n" + "="*60)
    print("SOURCE DISTRIBUTION (sorted by count)")
    print("="*60)

    for source, count in source_counter.most_common():
        percentage = 100 * count / len(all_sources)
        print(f"{source:50s}: {count:8,} ({percentage:5.2f}%)")

    # Save to file
    output_file = "temp/source_keywords_list.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {data_name}\n")
        f.write(f"Total samples analyzed: {len(all_sources):,}\n")
        f.write(f"Total unique sources: {len(unique_sources)}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("UNIQUE SOURCE KEYWORDS (alphabetical)\n")
        f.write("="*60 + "\n\n")

        for i, source in enumerate(unique_sources, 1):
            count = source_counter[source]
            f.write(f"{i:3d}. {source:50s} (count: {count:,})\n")

        f.write("\n" + "="*60 + "\n")
        f.write("SOURCE DISTRIBUTION (sorted by count)\n")
        f.write("="*60 + "\n\n")

        for source, count in source_counter.most_common():
            percentage = 100 * count / len(all_sources)
            f.write(f"{source:50s}: {count:8,} ({percentage:5.2f}%)\n")

    print(f"\nResults saved to: {output_file}")

    # Group samples by source and select 1 random example from each
    print("\n" + "="*60)
    print("SELECTING RANDOM EXAMPLES FROM EACH SOURCE")
    print("="*60)

    source_examples = defaultdict(list)
    for sample in all_samples:
        if 'source' in sample:
            source_examples[sample['source']].append(sample)

    # Select one random example from each source
    random_examples = {}
    for source in unique_sources:
        if source in source_examples and len(source_examples[source]) > 0:
            random_example = random.choice(source_examples[source])
            random_examples[source] = random_example
            print(f"Selected example from: {source}")

    # Save random examples to JSON
    examples_output_file = "temp/source_random_examples.json"
    with open(examples_output_file, 'w', encoding='utf-8') as f:
        json.dump(random_examples, f, indent=2, ensure_ascii=False)

    print(f"\n{len(random_examples)} random examples saved to: {examples_output_file}")
    print(f"Each source category has 1 random example in the JSON file.")

else:
    print("\nERROR: No source data found!")
    print("Please check if the dataset has a 'source' column.")