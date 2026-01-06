"""
Create a dataset with 100K samples from allenai/tulu-3-sft-mixture
Exclude samples from specific source categories.
"""

from datasets import load_dataset, Dataset
import json
import random

# Dataset to load
data_name = "allenai/tulu-3-sft-mixture"

# Categories to exclude
excluded_sources = [
    "ai2-adapt-dev/tulu_v3.9_aya_100k",
    "ai2-adapt-dev/oasst1_converted",
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
    "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
    "ai2-adapt-dev/tulu_hard_coded_repeated_10"
]

# Target number of samples
target_samples = 100000

print(f"Loading dataset: {data_name}")
print("This may take a while for large datasets...")

# Load the dataset
dataset = load_dataset(data_name)

print(f"\nDataset loaded successfully!")
print(f"Dataset structure: {dataset}")
print(f"\nAvailable splits: {list(dataset.keys())}")

print("\n" + "="*60)
print("FILTERING DATASET")
print("="*60)

print(f"\nExcluding the following source categories:")
for i, source in enumerate(excluded_sources, 1):
    print(f"  {i}. {source}")

# Collect all samples that are NOT in excluded sources (only process "train" split)
filtered_samples = []

split_data = dataset['train']

for sample in split_data:
    if sample['source'] not in excluded_sources:
        filtered_samples.append(sample)

print("\n" + "="*60)
print("FILTERING RESULTS")
print("="*60)

print(f"\nTotal samples after filtering: {len(filtered_samples):,}")
print(f"Target samples: {target_samples:,}")

# Group samples by source for uniform sampling
from collections import defaultdict
source_groups = defaultdict(list)
for sample in filtered_samples:
    source_groups[sample['source']].append(sample)

num_sources = len(source_groups)
print(f"\nNumber of unique sources: {num_sources}")

# Calculate samples per source for uniform distribution
samples_per_source = target_samples // num_sources
print(f"Target samples per source (uniform): {samples_per_source}")

# Perform uniform sampling
final_samples = []
remaining_quota = target_samples

for source, samples in source_groups.items():
    # Take minimum of: samples_per_source, available samples, or remaining quota
    num_to_take = min(samples_per_source, len(samples), remaining_quota)

    # Randomly sample from this source
    if len(samples) <= num_to_take:
        # Take all samples if we don't have enough
        selected = samples
    else:
        # Randomly sample if we have more than needed
        selected = random.sample(samples, num_to_take)

    final_samples.extend(selected)
    remaining_quota -= len(selected)

# If we still have quota remaining, fill it with random samples from sources that have extras
if remaining_quota > 0 and len(final_samples) < target_samples:
    print(f"\nFilling remaining {remaining_quota} samples from sources with extras...")

    # Get sources that have more samples than we took
    extra_samples = []
    for source, samples in source_groups.items():
        if len(samples) > samples_per_source:
            # Get samples we didn't take yet
            taken_ids = {id(s) for s in final_samples if s['source'] == source}
            available = [s for s in samples if id(s) not in taken_ids]
            extra_samples.extend(available)

    if extra_samples:
        num_extra_to_take = min(remaining_quota, len(extra_samples))
        final_samples.extend(random.sample(extra_samples, num_extra_to_take))

print(f"\nSelected {len(final_samples):,} samples total")

# Show source distribution in final dataset
print("\n" + "="*60)
print("SOURCE DISTRIBUTION IN FINAL DATASET")
print("="*60)

from collections import Counter
source_counter = Counter([sample['source'] for sample in final_samples if 'source' in sample])

print(f"\nTotal unique sources: {len(source_counter)}")
print(f"\nTop 20 sources by count:")
for i, (source, count) in enumerate(source_counter.most_common(20), 1):
    percentage = 100 * count / len(final_samples)
    print(f"{i:2d}. {source:60s}: {count:8,} ({percentage:5.2f}%)")

# Convert to the required format (instruction, input, output)
print(f"\n" + "="*60)
print("CONVERTING TO REQUIRED FORMAT")
print("="*60)

converted_samples = []
conversion_errors = 0

for i, sample in enumerate(final_samples):
    try:
        if 'messages' in sample and isinstance(sample['messages'], list):
            messages = sample['messages']

            # Find user and assistant messages
            user_content = ""
            assistant_content = ""

            for msg in messages:
                if msg.get('role') == 'user':
                    user_content = msg.get('content', '')
                elif msg.get('role') == 'assistant':
                    assistant_content = msg.get('content', '')

            # Create the converted format
            converted_sample = {
                "instruction": "",  # Left blank as requested
                "input": user_content,
                "output": assistant_content
            }
            converted_samples.append(converted_sample)
        else:
            conversion_errors += 1
            print(f"Warning: Sample {i} does not have 'messages' field or it's not a list")
    except Exception as e:
        conversion_errors += 1
        print(f"Error converting sample {i}: {e}")

print(f"\nSuccessfully converted: {len(converted_samples):,} samples")
print(f"Conversion errors: {conversion_errors}")

# Save to JSON file in the new format
output_file = "data/tulu_100k.json"
print(f"\n" + "="*60)
print("SAVING DATASET")
print("="*60)

print(f"\nSaving {len(converted_samples):,} samples to: {output_file}")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_samples, f, indent=2, ensure_ascii=False)

print(f"Dataset saved successfully!")

# Save metadata
metadata_file = "data/tulu_filtered_100k_metadata.txt"
with open(metadata_file, 'w', encoding='utf-8') as f:
    f.write(f"Dataset: {data_name}\n")
    f.write(f"Created filtered dataset with {len(final_samples):,} samples\n")
    f.write(f"Target samples: {target_samples:,}\n")
    f.write(f"\nExcluded sources:\n")
    for source in excluded_sources:
        f.write(f"  - {source}\n")
    f.write(f"\nTotal unique sources in final dataset: {len(source_counter)}\n")
    f.write(f"\nSource distribution:\n")
    for source, count in source_counter.most_common():
        percentage = 100 * count / len(final_samples)
        f.write(f"{source:60s}: {count:8,} ({percentage:5.2f}%)\n")

print(f"Metadata saved to: {metadata_file}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
print(f"\nFinal dataset: {output_file}")
print(f"Samples: {len(final_samples):,}")
print(f"Unique sources: {len(source_counter)}")
