#!/usr/bin/env python3
"""
Utility script to preprocess a dataset for LoRWeB training.
This script creates training datasets where control images are concatenated triplets (a, a', b) and targets are b' images
such that the analogy a:a' :: b:b' holds.
"""

import json
import argparse
import random
from pathlib import Path
from PIL import Image
from PIL import Image as PILImage
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class Relation252KDataset:
    """Standalone dataset loader for Relation252K dataset."""

    def __init__(
        self,
        data_dir: str,
        split_images: bool = True,
        include_category: bool = True,
        max_samples_per_category: Optional[int] = None,
        categories_filter: Optional[List[str]] = None,
    ):
        """
        Initialize the Relation252K dataset.

        Args:
            data_dir: Path to the dataset directory (HuggingFace cache snapshot)
            split_images: Whether to split concatenated images into left and right parts
            include_category: Whether to include category information in the dataset
            max_samples_per_category: Maximum number of samples to load per category
            categories_filter: List of category names to include (None means all)
        """
        self.data_dir = data_dir
        self.split_images = split_images
        self.include_category = include_category
        self.max_samples_per_category = max_samples_per_category
        self.categories_filter = categories_filter

        # Load all samples
        self.samples = self._load_all_samples()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index."""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        return self.samples[idx]

    def __iter__(self):
        """Iterate over all samples."""
        for sample in self.samples:
            yield sample

    def _find_categories(self, data_dir: str) -> List[Tuple[str, str]]:
        """
        Find all categories and their Group1 directories.

        Returns:
            List of tuples (category_name, group1_path)
        """
        categories = []
        data_path = Path(data_dir)

        # Look for directories that contain Group1 subdirectories
        for item in tqdm(data_path.iterdir(), total=len(list(data_path.iterdir()))):
            if item.is_dir():
                group1_path = item / "Group1"
                if group1_path.exists() and group1_path.is_dir():
                    # Check if labels.json exists
                    labels_path = group1_path / "labels.json"
                    if labels_path.exists():
                        category_name = item.name

                        # Apply category filter if specified
                        if (self.categories_filter is None or
                            category_name in self.categories_filter):
                            categories.append((category_name, str(group1_path)))
        return categories

    def _load_labels(self, labels_path: str) -> List[Dict[str, Any]]:
        """Load and parse the labels.json file."""
        with open(labels_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _split_concatenated_image(self, image_path: str) -> Tuple[PILImage.Image, PILImage.Image]:
        """
        Split a horizontally concatenated image into left and right parts.

        Args:
            image_path: Path to the concatenated image

        Returns:
            Tuple of (left_image, right_image)
        """
        image = PILImage.open(image_path)
        width, height = image.size

        # Split horizontally in the middle
        left_image = image.crop((0, 0, width // 2, height))
        right_image = image.crop((width // 2, 0, width, height))

        return left_image, right_image

    def _load_all_samples(self) -> List[Dict[str, Any]]:
        """Load all samples from the dataset."""
        categories = self._find_categories(self.data_dir)
        samples = []

        for category_name, group1_path in tqdm(categories, total=len(categories), desc="loading all samples"):
            # Load labels for this category
            labels_path = os.path.join(group1_path, "labels.json")
            try:
                labels_data = self._load_labels(labels_path)
            except Exception as e:
                print(f"Error loading labels for {category_name}: {e}")
                continue

            # Apply max samples limit if specified
            if self.max_samples_per_category:
                labels_data = labels_data[:self.max_samples_per_category]

            for label_entry in tqdm(labels_data, total=len(labels_data), desc=f"loading all samples for {category_name}"):
                img_name = label_entry["img_name"]
                image_path = os.path.join(group1_path, img_name)

                # Skip if image file doesn't exist
                if not os.path.exists(image_path):
                    continue

                # Create unique image ID
                image_id = f"{category_name}_{img_name}"

                # Prepare the sample data
                sample = {
                    "image_id": image_id,
                    "left_image_description": label_entry["left_image_description"],
                    "right_image_description": label_entry["right_image_description"],
                    "edit_instruction": label_entry["edit_instruction"],
                    "img_name": img_name,
                }

                # Add category if requested
                if self.include_category:
                    sample["category"] = category_name

                # Handle image loading based on configuration
                if self.split_images:
                    try:
                        left_img, right_img = self._split_concatenated_image(image_path)
                        sample["original_image"] = left_img
                        sample["transformed_image"] = right_img
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                        continue
                else:
                    # Keep concatenated image
                    try:
                        concatenated_img = PILImage.open(image_path)
                        sample["concatenated_image"] = concatenated_img
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        continue

                samples.append(sample)

        return samples


def process_single_analogy(args):
    """
    Process a single analogy in a separate process.

    Args:
        args: Tuple containing (analogy_data, i, target_path, control_path, output_path, existing_analogy_ids)

    Returns:
        Tuple of (success, metadata_entry or None, error_message or None)
    """
    analogy, i, target_path, control_path, output_path = args

    try:
        # Create unique filename for this analogy
        direction_suffix = "_rev" if analogy['direction'] == 'reverse' else ""
        analogy_id = f"analogy_{analogy['A_id']}_{analogy['C_id']}_{i:04d}{direction_suffix}"

        # Check if this analogy already exists and skip if so
        #if analogy_id in existing_analogy_ids:
        #    return True, None, None  # Skip but don't count as error

        control_output_path = control_path / f"{analogy_id}.jpg"
        target_image_path = target_path / f"{analogy_id}.jpg"

        # Also check if files already exist on disk
        #if control_output_path.exists() and target_image_path.exists():
        #    return True, None, None  # Skip but don't count as error

        # Create concatenated control image: A | B | C (PIL Images)
        control_images = [
            analogy['A_source'],  # A: original image (PIL)
            analogy['A_edited'],  # B: edited version of A (PIL)
            analogy['C_source']   # C: another original image (PIL)
        ]

        # Create concatenated image (inline the logic to avoid self reference)
        if len(control_images) != 3:
            return False, None, f"Expected 3 images, got {len(control_images)}"

        # Resize all images to the same size (use the smallest dimensions)
        min_width = min(img.width for img in control_images)
        min_height = min(img.height for img in control_images)

        # Convert to numpy arrays and resize in one go (much faster than PIL paste)
        arrays = []
        for img in control_images:
            if img.size != (min_width, min_height):
                img = img.resize((min_width, min_height), Image.Resampling.LANCZOS)

            # Convert RGBA to RGB if necessary to avoid channel mismatch
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode not in ['RGB', 'L']:  # Handle other modes
                img = img.convert('RGB')

            arrays.append(np.array(img))

        # Ensure all arrays have the same number of channels
        target_channels = arrays[0].shape[2] if len(arrays[0].shape) == 3 else 1
        normalized_arrays = []
        for arr in arrays:
            if len(arr.shape) == 3 and arr.shape[2] != target_channels:
                # Convert to PIL, then to target mode, then back to array
                temp_img = Image.fromarray(arr)
                if target_channels == 3:
                    temp_img = temp_img.convert('RGB')
                elif target_channels == 1:
                    temp_img = temp_img.convert('L')
                arr = np.array(temp_img)
            normalized_arrays.append(arr)

        # Concatenate horizontally using numpy (5-10x faster than PIL paste operations)
        concatenated_array = np.concatenate(normalized_arrays, axis=1)

        # Convert back to PIL and save
        concatenated = Image.fromarray(concatenated_array)
        # Ensure RGB mode for JPEG saving
        if concatenated.mode == 'RGBA':
            concatenated = concatenated.convert('RGB')
        concatenated.save(control_output_path, "JPEG", quality=95)

        # Target image is D: edited version of C (PIL Image)
        target_image = analogy['C_edited']  # PIL Image
        # Ensure RGB mode for JPEG saving
        if target_image.mode == 'RGBA':
            target_image = target_image.convert('RGB')
        target_image.save(target_image_path, "JPEG", quality=95)

        # Create caption file explaining the analogy
        caption_path = target_path / f"{analogy_id}.txt"
        with open(caption_path, 'w') as f:
            f.write(analogy['A_edit_prompt'])

        # Collect metadata for JSON output
        metadata_entry = {
            'analogy_id': analogy_id,
            'edit_prompt': analogy['A_edit_prompt'],
            'category': analogy['category'],
            'direction': analogy['direction'],
            'control_image_path': str(control_output_path.relative_to(output_path)),
            'target_image_path': str(target_image_path.relative_to(output_path)),
            'caption_path': str(caption_path.relative_to(output_path)),
            # Pair A metadata
            'pair_A': {
                'image_id': analogy['A_id'],
                'edit_prompt': analogy['A_edit_prompt'],
                'left_image_description': analogy['A_left_description'],
                'right_image_description': analogy['A_right_description'],
                'img_name': analogy['A_img_name']
            },
            # Pair C metadata
            'pair_C': {
                'image_id': analogy['C_id'],
                'edit_prompt': analogy['C_edit_prompt'],
                'left_image_description': analogy['C_left_description'],
                'right_image_description': analogy['C_right_description'],
                'img_name': analogy['C_img_name']
            }
        }

        return True, metadata_entry, None

    except Exception as e:
        return False, None, str(e)


class KontextAnalogyDatasetCreator:
    def __init__(self, relation_dataset_path):
        self.relation_dataset = Relation252KDataset(
            data_dir=relation_dataset_path,
            split_images=True,  # We need split images for analogies
            include_category=True,
            max_samples_per_category=None  # Load all samples initially
        )

    def get_triplets(self, filter_json=None, max_samples_per_category=None):
        """Get all triplets from Relation252K dataset format"""
        triplets = []
        filtered_images = set()

        # Load filtered images if filter_json is provided
        if filter_json:
            with open(filter_json, 'r') as f:
                filter_data = json.load(f)
                # Create a set of identifiers to ignore
                for item in filter_data:
                    filtered_images.add(item.get('image_id', ''))
                print(f"Loaded {len(filtered_images)} filtered image pairs to ignore")

        # Group samples by category and limit if requested
        category_counts = defaultdict(int)

        for sample in self.relation_dataset:
            # Skip if filtered
            if sample['image_id'] in filtered_images:
                continue

            # Apply category limits
            category = sample.get('category', 'unknown')
            if max_samples_per_category and category_counts[category] >= max_samples_per_category:
                continue
            category_counts[category] += 1

            # Create triplet in the expected format
            triplet = {
                'source_image': sample['original_image'],  # PIL Image
                'edited_image': sample['transformed_image'],  # PIL Image
                'edit_prompt': sample['edit_instruction'],
                'image_id': sample['image_id'],
                'category': sample.get('category', 'unknown'),
                'left_image_description': sample['left_image_description'],
                'right_image_description': sample['right_image_description'],
                'img_name': sample['img_name']
            }
            triplets.append(triplet)

        if filter_json:
            print(f"Filtered out {len(filtered_images)} image pairs, {len(triplets)} remaining")

        print(f"Loaded {len(triplets)} triplets from {len(category_counts)} categories")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {category}: {count} samples")

        return triplets

    def group_by_category(self, triplets):
        """Group triplets by category to find pairs for analogies"""
        category_groups = defaultdict(list)
        for triplet in triplets:
            category_groups[triplet['category']].append(triplet)
        return category_groups

    def create_concatenated_image(self, images, output_path, layout='horizontal'):
        """
        Create a concatenated image from multiple image paths.

        Args:
            images: List of PIL Images to concatenate
            output_path: Path to save the concatenated image
            layout: 'horizontal' or 'vertical' concatenation
        """

        if len(images) != 3:
            print(f"Error: Expected 3 images, got {len(images)}")
            return False

        # Resize all images to the same size (use the smallest dimensions)
        min_width = min(img.width for img in images)
        min_height = min(img.height for img in images)

        resized_images = []
        for img in images:
            resized = img.resize((min_width, min_height), Image.Resampling.LANCZOS)
            resized_images.append(resized)

        # Concatenate horizontally: A | B | C
        if layout == 'horizontal':
            total_width = min_width * 3
            total_height = min_height
            concatenated = Image.new('RGB', (total_width, total_height))

            for i, img in enumerate(resized_images):
                x_offset = i * min_width
                concatenated.paste(img, (x_offset, 0))

        # Concatenate vertically: A above B above C
        elif layout == 'vertical':
            total_width = min_width
            total_height = min_height * 3
            concatenated = Image.new('RGB', (total_width, total_height))

            for i, img in enumerate(resized_images):
                y_offset = i * min_height
                concatenated.paste(img, (0, y_offset))

        concatenated.save(output_path, "JPEG", quality=95)
        return True

    def create_analogy_dataset(self, output_dir, max_analogies=None, filter_categories=None,
                               filter_json=None, min_pairs_per_category=2,
                               max_samples_per_category=None, num_processes=None):
        """
        Create a LoRWeB analogy training dataset from the Relation252K dataset.

        Args:
            output_dir: Directory to create the training dataset
            max_analogies: Maximum number of analogies to include
            filter_categories: List of categories to include (Relation252K category names)
            filter_json: JSON file to filter out specific images
            min_pairs_per_category: Minimum number of pairs needed per category to create analogies
            max_samples_per_category: Maximum samples to load per category
            num_processes: Number of processes to use for parallel processing (None = auto-detect)
        """
        triplets = self.get_triplets(filter_json, max_samples_per_category)

        # Filter by categories if specified (now using Relation252K category names)
        if filter_categories:
            filtered_triplets = []
            for triplet in triplets:
                category = triplet.get('category', '')
                if category in filter_categories:
                    filtered_triplets.append(triplet)
            triplets = filtered_triplets
            print(f"Filtered to {len(triplets)} triplets from categories: {filter_categories}")

        # Group triplets by category
        category_groups = self.group_by_category(triplets)

        # Filter categories that have enough pairs for analogies
        valid_categories = {}
        for category, group in category_groups.items():
            if len(group) >= min_pairs_per_category:
                valid_categories[category] = group

        print(f"Found {len(valid_categories)} categories with at least {min_pairs_per_category} pairs each")

        # Create analogies with rich metadata
        analogies = []
        for category, group in tqdm(valid_categories.items(), total=len(valid_categories), desc="creating all analogies"):
            # Create all possible pairs for this prompt
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # Original analogy: AA'B → B (where A' is edited A, B is original C, target is edited C)
                    analogy = {
                        'category': category,  # Add this line
                        'A_edit_prompt': group[i]['edit_prompt'],  # Keep this for the caption
                        'A_source': group[i]['source_image'],  # PIL Image
                        'A_edited': group[i]['edited_image'],  # PIL Image
                        'C_edit_prompt': group[j]['edit_prompt'],  # Keep this for the caption
                        'C_source': group[j]['source_image'],  # PIL Image
                        'C_edited': group[j]['edited_image'],  # PIL Image
                        'A_id': group[i]['image_id'],
                        'C_id': group[j]['image_id'],
                        'direction': 'forward',  # AA'B → B
                        # Rich metadata for pair A
                        'A_left_description': group[i]['left_image_description'],
                        'A_right_description': group[i]['right_image_description'],
                        'A_img_name': group[i]['img_name'],
                        # Rich metadata for pair C
                        'C_left_description': group[j]['left_image_description'],
                        'C_right_description': group[j]['right_image_description'],
                        'C_img_name': group[j]['img_name'],
                    }
                    analogies.append(analogy)

        # Limit analogies if specified
        if max_analogies and len(analogies) > max_analogies:
            analogies = random.sample(analogies, max_analogies)

        # Create output directory structure
        output_path = Path(output_dir)
        target_path = output_path / "target"
        control_path = output_path / "control"

        target_path.mkdir(parents=True, exist_ok=True)
        control_path.mkdir(parents=True, exist_ok=True)

        print(f"Creating analogy training dataset with {len(analogies)} samples...")
        print(f"Target images: {target_path}")
        print(f"Control images: {control_path}")

        # Process each analogy using multiprocessing and collect metadata for JSON output
        successful_analogies = 0
        analogy_metadata = []
        existing_analogy_ids = set()

        # Load existing metadata if it exists to continue from where we left off
        metadata_json_path = output_path / "analogy_metadata.json"
        if metadata_json_path.exists():
            try:
                with open(metadata_json_path, 'r') as f:
                    existing_metadata = json.load(f)
                    analogy_metadata = existing_metadata
                    existing_analogy_ids = {entry['analogy_id'] for entry in existing_metadata}
                    successful_analogies = len(existing_metadata)
                print(f"Loaded {len(existing_metadata)} existing analogies from metadata file")
            except Exception as e:
                print(f"Warning: Could not load existing metadata: {e}")
                print("Starting fresh...")

        # Determine number of processes
        if num_processes is None:
            num_processes = min(mp.cpu_count(), len(analogies))

        print(f"Using {num_processes} processes for parallel analogy processing...")

        # Prepare arguments for multiprocessing
        process_args = [(analogy, i, target_path, control_path, output_path)
                for i, analogy in enumerate(analogies) if f"analogy_{analogy['A_id']}_{analogy['C_id']}_{i:04d}" not in existing_analogy_ids]

        # Process analogies in parallel
        with mp.Pool(processes=num_processes) as pool:
            # Use imap for progress tracking
            results = list(tqdm(
                pool.imap(process_single_analogy, process_args),
                total=len(process_args),
                desc="saving analogies"
            ))

        # Collect results and handle errors
        failed_count = 0
        skipped_count = 0
        new_analogies_count = 0
        for i, (success, metadata_entry, error_msg) in enumerate(results):
            if success:
                if metadata_entry:
                    analogy_metadata.append(metadata_entry)
                    new_analogies_count += 1
                    successful_analogies += 1
                else:
                    # This was skipped (already exists)
                    skipped_count += 1
            else:
                failed_count += 1
                if error_msg:
                    print(f"Error processing analogy {i}: {error_msg}")

        print(f"Processing complete: {new_analogies_count} new, {skipped_count} skipped, {failed_count} failed")

        if failed_count > 0:
            print(f"Failed to process {failed_count} analogies out of {len(analogies)}")

        # Save comprehensive JSON metadata (overwrite with complete metadata)
        metadata_json_path = output_path / "analogy_metadata.json"
        with open(metadata_json_path, 'w') as f:
            json.dump(analogy_metadata, f, indent=2, ensure_ascii=False)

        print(f"Saved metadata for {len(analogy_metadata)} total analogies to {metadata_json_path}")

        # Create a summary file
        summary_path = output_path / "analogy_dataset_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"LoRWeB Analogy Training Dataset Summary (Relation252K)\n")
            f.write(f"==========================================================\n\n")
            f.write(f"Total analogies: {successful_analogies}\n")
            f.write(f"Target folder: {target_path}\n")
            f.write(f"Control folder: {control_path}\n")
            f.write(f"Metadata JSON: {metadata_json_path}\n\n")

            # Prompt breakdown
            category_counts = defaultdict(int)
            for entry in analogy_metadata:
                category_counts['category'] += 1

            f.write(f"\nCategory breakdown:\n")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {category}: {count}\n")

            f.write(f"\nAnalogy structure: A (original) | B (edited A) | C (original) -> D (edited C)\n")
            f.write(f"Learning goal: A:B :: C:D\n")
            f.write(f"\nMetadata includes:\n")
            f.write(f"- Rich image descriptions for both pairs\n")
            f.write(f"- Category information\n")
            f.write(f"- Edit instructions\n")
            f.write(f"- File paths and identifiers\n")

        print(f"\nAnalogy training dataset created successfully!")
        print(f"Output directory: {output_path}")
        print(f"Target images: {target_path}")
        print(f"Control images: {control_path}")
        print(f"Summary: {summary_path}")
        print(f"Metadata JSON: {metadata_json_path}")
        print(f"Successful analogies: {successful_analogies}")

        return output_path

def main():
    parser = argparse.ArgumentParser(description="Create LoRWeB analogy training dataset from Relation252K dataset")
    parser.add_argument("--relation_dataset_path", type=str, required=True,
                       help="Path to the Relation252K dataset directory")
    parser.add_argument("--output_path", type=str, default="./data/relation252k_processed",
                       help="Output path for dataset creation")
    parser.add_argument("--max_analogies", type=int, default=None,
                       help="Maximum number of analogies to include in training dataset")
    parser.add_argument("--filter_categories", nargs='+', default=None,
                       help="Filter by specific Relation252K categories (e.g., '115-Colored Pencil Filter')")
    parser.add_argument("--filter_json", type=str,
                       help="Filter specific images from a JSON file")
    parser.add_argument("--min_pairs_per_category", type=int, default=2,
                       help="Minimum number of pairs needed per category to create analogies")
    parser.add_argument("--max_samples_per_category", type=int, default=None,
                       help="Maximum samples to load per category (for faster testing)")
    parser.add_argument("--num_processes", type=int, default=None,
                       help="Number of processes to use for parallel processing (default: auto-detect)")

    args = parser.parse_args()

    creator = KontextAnalogyDatasetCreator(args.relation_dataset_path)

    creator.create_analogy_dataset(
        args.output_path,
        max_analogies=args.max_analogies,
        filter_categories=args.filter_categories,
        filter_json=args.filter_json,
        min_pairs_per_category=args.min_pairs_per_category,
        max_samples_per_category=args.max_samples_per_category,
        num_processes=args.num_processes
    )


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    main()
