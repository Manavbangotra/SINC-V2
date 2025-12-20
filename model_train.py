import argparse
import os
import random
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import time
import os
from datetime import datetime
from pathlib import Path

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    SiglipModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report
from model_arch import MultimodalClassifier

# -------------------------
# Image Validation Functions
# -------------------------
def is_valid_image(url, min_size=(32, 32), max_size=(10000, 10000), variance_threshold=10):
    """
    Validates if an image URL points to a valid, non-blank image.
    
    Args:
        url: Image URL to validate
        min_size: Minimum (width, height) dimensions
        max_size: Maximum (width, height) dimensions
        variance_threshold: Minimum pixel variance to consider image non-blank
    
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        # Download image with timeout
        response = requests.get(url, timeout=10, stream=True)
        if response.status_code != 200:
            return False
        
        # Open and verify it's a valid image
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Check dimensions
        width, height = img.size
        if width < min_size[0] or height < min_size[1]:
            return False
        if width > max_size[0] or height > max_size[1]:
            return False
        
        # Check if image is blank (all same color or very low variance)
        img_array = np.array(img)
        
        # Calculate variance for each channel
        variances = [np.var(img_array[:, :, i]) for i in range(3)]
        max_variance = max(variances)
        
        # If variance is too low, image is likely blank/solid color
        if max_variance < variance_threshold:
            return False
        
        # Additional check: ensure image is not all white or all black
        mean_pixel = np.mean(img_array)
        if mean_pixel > 250 or mean_pixel < 5:
            # Check if it's truly uniform (low std dev)
            std_dev = np.std(img_array)
            if std_dev < 5:
                return False
        
        return True
        
    except Exception as e:
        # Any error means invalid image
        return False

async def is_valid_image_async(session, url, min_size=(32, 32), max_size=(10000, 10000), variance_threshold=10):
    """
    Async version of image validation for batch processing.
    
    Args:
        session: aiohttp ClientSession
        url: Image URL to validate
        min_size: Minimum (width, height) dimensions
        max_size: Maximum (width, height) dimensions
        variance_threshold: Minimum pixel variance to consider image non-blank
    
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status != 200:
                return False
            
            content = await response.read()
            img = Image.open(BytesIO(content))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Check dimensions
            width, height = img.size
            if width < min_size[0] or height < min_size[1]:
                return False
            if width > max_size[0] or height > max_size[1]:
                return False
            
            # Check if image is blank (all same color or very low variance)
            img_array = np.array(img)
            
            # Calculate variance for each channel
            variances = [np.var(img_array[:, :, i]) for i in range(3)]
            max_variance = max(variances)
            
            # If variance is too low, image is likely blank/solid color
            if max_variance < variance_threshold:
                return False
            
            # Additional check: ensure image is not all white or all black
            mean_pixel = np.mean(img_array)
            if mean_pixel > 250 or mean_pixel < 5:
                # Check if it's truly uniform (low std dev)
                std_dev = np.std(img_array)
                if std_dev < 5:
                    return False
            
            return True
            
    except Exception as e:
        # Any error means invalid image
        return False

def filter_valid_images(df, image_url_column='src', use_async=True, batch_size=100):
    """
    Filters DataFrame to keep only rows with valid images.
    
    Args:
        df: pandas DataFrame with image URLs
        image_url_column: Name of column containing image URLs
        use_async: Whether to use async processing (faster for large datasets)
        batch_size: Batch size for async processing
    
    Returns:
        pandas DataFrame with only valid images
    """
    import pandas as pd
    
    if use_async:
        import aiohttp
        import asyncio
        
        async def validate_batch(urls):
            timeout = aiohttp.ClientTimeout(total=10)
            connector = aiohttp.TCPConnector(limit=50)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                tasks = [is_valid_image_async(session, url) for url in urls]
                return await asyncio.gather(*tasks)
        
        valid_mask = []
        urls = df[image_url_column].tolist()
        
        print(f"Validating {len(urls)} images...")
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i+batch_size]
            batch_results = asyncio.run(validate_batch(batch_urls))
            valid_mask.extend(batch_results)
            print(f"Processed {min(i+batch_size, len(urls))}/{len(urls)} images")
        
        valid_mask = pd.Series(valid_mask, index=df.index)
    else:
        print(f"Validating {len(df)} images (this may take a while)...")
        valid_mask = df[image_url_column].apply(is_valid_image)
        print("Validation complete!")
    
    valid_count = valid_mask.sum()
    invalid_count = len(df) - valid_count
    print(f"Valid images: {valid_count}, Invalid images: {invalid_count}")
    
    return df[valid_mask].reset_index(drop=True)

# -------------------------
# Config / Arguments
# -------------------------
parser = argparse.ArgumentParser()
# Data arguments
parser.add_argument("--csv", type=str, default="all_products50.csv", help="Merged CSV path")
parser.add_argument("--text_model", type=str, default="microsoft/deberta-v3-small")
parser.add_argument("--image_model", type=str, default="google/siglip2-base-patch16-256")
parser.add_argument("--max_length", type=int, default=64)
parser.add_argument("--sample", type=int, default=0, help="If >0, sample this many rows for quick runs")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--chunk_size", type=int, default=700000, help="Process dataset in chunks to limit disk usage")

# Training arguments
parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
parser.add_argument("--gradient_accumulation_steps", type=int, default=32, 
                    help="Number of steps to accumulate gradients before optimizer step")
parser.add_argument("--epochs", type=int, default=10, 
                    help="Number of training epochs (recommended: 5 for full training schedule)")

# Learning rates (differential learning rates for different components)
parser.add_argument("--lr_mlp", type=float, default=1e-3, help="Learning rate for MLP head")
parser.add_argument("--lr_projection", type=float, default=3e-4, 
                    help="Learning rate for projection layers")
parser.add_argument("--lr_text", type=float, default=1e-5, 
                    help="Learning rate for text encoder (DeBERTa)")
parser.add_argument("--lr_image", type=float, default=5e-6, 
                    help="Learning rate for image encoder (SigLIP)")

# Checkpointing
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                    help="Directory to save checkpoints")
parser.add_argument("--checkpoint_interval", type=int, default=36000, 
                    help="Save checkpoint every N seconds (default: 10 hours)")
parser.add_argument("--resume_from", type=str, default="", 
                    help="Path to checkpoint to resume training from")

# Progressive unfreezing
parser.add_argument("--unfreeze_text_layers", type=int, default=0, 
                    help="Number of text encoder layers to unfreeze (0 = frozen). "
                         "For SigLIP, this is the percentage of layers to unfreeze (1-100).")
parser.add_argument("--unfreeze_image_layers", type=int, default=0, 
                    help="For SigLIP, this is the percentage of vision layers to unfreeze (1-100).")

# Parse arguments - only when script is run directly, not when imported
if __name__ == "__main__":
    args = parser.parse_args()
else:
    # Being imported as module - create default args object to avoid errors
    class DefaultArgs:
        # csv = "all_products.csv"
        csv = "all_products50.csv"
        text_model = "microsoft/deberta-v3-small"
        image_model = "google/siglip2-base-patch16-256"
        max_length = 64
        batch_size = 8
        gradient_accumulation_steps = 32
        epochs = 3
        lr_mlp = 1e-3
        lr_projection = 3e-4
        lr_text = 1e-5
        lr_image = 5e-6
        sample = 0
        seed = 42
        chunk_size = 700000
        checkpoint_dir = "checkpoints"
        checkpoint_interval = 36000
        resume_from = ""
        unfreeze_text_layers = 0
        unfreeze_image_layers = 0
    args = DefaultArgs()

# Only run training code when script is executed directly
if __name__ == "__main__":
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------------------------
    # 1) Load CSV(s) with datasets
    # -------------------------
    print("Loading CSV:", args.csv)
    ds = load_dataset("csv", data_files=args.csv, split="train")
    ds = ds.filter(lambda x: x["classified_niche_ai"] is not None)
    # If you want a small quick sample (useful for debugging)
    if args.sample and args.sample > 0:
        print(f"Sampling {args.sample} rows for quick run")
        ds = ds.shuffle(seed=args.seed).select(range(min(args.sample, len(ds))))

    # Ensure columns expected: title, src (image url), niche
    expected = ["title", "src", "classified_niche_ai"]
    for c in expected:
        if c not in ds.column_names:
            raise ValueError(f"CSV must contain column '{c}'. Found: {ds.column_names}")

    # Encode labels
    labels = sorted(list(set(ds["classified_niche_ai"])))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    def add_label(example):
        example["label"] = label2id[example["classified_niche_ai"]]
        return example

    ds = ds.map(add_label)

    # Stratified split: train/val/test = 80/10/10
    print("Creating train/val/test splits (stratified)...")
    def stratified_split(dataset, label_col="label", train_frac=0.8, val_frac=0.1, seed=42):
        # naive stratified split implementation: group by label and split each group
        indices_by_label = {}
        for i, lbl in enumerate(dataset[label_col]):
            indices_by_label.setdefault(lbl, []).append(i)
        train_idx, val_idx, test_idx = [], [], []
        rng = random.Random(seed)
        for lbl, idxs in indices_by_label.items():
            rng.shuffle(idxs)
            n = len(idxs)
            n_train = int(n * train_frac)
            n_val = int(n * val_frac)
            train_idx.extend(idxs[:n_train])
            val_idx.extend(idxs[n_train:n_train+n_val])
            test_idx.extend(idxs[n_train+n_val:])
        return (
            dataset.select(train_idx),
            dataset.select(val_idx),
            dataset.select(test_idx),
        )

    train_ds, val_ds, test_ds = stratified_split(ds, label_col="label", seed=args.seed)
    print("Sizes:", len(train_ds), len(val_ds), len(test_ds))

    # -------------------------
    # 2) Tokenizers and processors
    # -------------------------
    print("Loading tokenizers / processors...")
    # DeBERTa uses SentencePiece -> force slow tokenizer
    # text_tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=False)
    # text_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
    clip_processor = AutoProcessor.from_pretrained(args.image_model)
    clip_model = SiglipModel.from_pretrained(args.image_model).to(device)  # used for image features

    # Optionally load text model separately (we'll load inside the model class)
    text_model_name = args.text_model
    image_model_name = args.image_model

    import os
    import aiohttp
    import asyncio
    from aiohttp import ClientTimeout
    from io import BytesIO
    from PIL import Image
    from datasets import load_dataset

    SAVE_DIR = "images"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------------------------
    # Async fetch function
    # -------------------------
    async def fetch_image(session, url, idx):
        save_path = os.path.join(SAVE_DIR, f"{idx}.jpg")
        if os.path.exists(save_path):
            return  # skip if already downloaded

        try:
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    img = Image.open(BytesIO(content)).convert("RGB")
                    img.save(save_path, "JPEG")
                else:
                    Image.new("RGB", (224, 224), (255, 255, 255)).save(save_path, "JPEG")
        except:
            Image.new("RGB", (224, 224), (255, 255, 255)).save(save_path, "JPEG")

    # -------------------------
    # Process one batch
    # -------------------------
    async def process_batch(urls, start_idx=0):
        timeout = ClientTimeout(total=10)
        connector = aiohttp.TCPConnector(limit=200)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = []
            for i, url in enumerate(urls):
                idx = start_idx + i
                tasks.append(fetch_image(session, url, idx))
            await asyncio.gather(*tasks)

    # -------------------------
    # Download in 10k chunks
    # -------------------------
    def download_in_chunks(ds, chunk_size=10000):
        urls = ds["src"]  # column name must match your dataset
        for start in range(0, len(urls), chunk_size):
            end = min(start + chunk_size, len(urls))
            print(f"Downloading {start} → {end} ...")
            asyncio.run(process_batch(urls[start:end], start_idx=start))

    # -------------------------
    # Preprocessing (use local files)
    # -------------------------
    def preprocess_example(example, idx):
        # Text encoding
        enc = text_tokenizer(
            example["title"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_attention_mask=True,
        )

        # Store image path only; load/transform in collate_fn to avoid huge Arrow arrays
        img_path = os.path.join(SAVE_DIR, f"{idx}.jpg")
        example["input_ids"] = enc["input_ids"]
        example["attention_mask"] = enc["attention_mask"]
        example["image_path"] = img_path
        return example

    # -------------------------
    # Chunked processing helpers (to cap disk usage)
    # -------------------------

    def iter_chunk_ranges(n, chunk_size):
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield start, end

    def download_range(urls, offset, network_batch=10000):
        for start in range(0, len(urls), network_batch):
            end = min(start + network_batch, len(urls))
            print(f"Downloading {offset + start} → {offset + end} ...")
            asyncio.run(process_batch(urls[start:end], start_idx=offset + start))

    def preprocess_example_with_offset(example, idx, offset):
        return preprocess_example(example, idx + offset)

    def cleanup_images_range(start, end):
        for idx in range(start, end):
            p = os.path.join(SAVE_DIR, f"{idx}.jpg")
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    # Convert to torch format at iteration time via collate_fn
    # -------------------------
    # 4) DataLoader collate_fn
    # -------------------------
    import torch.nn.functional as F

    def collate_fn(batch):
        # Process text inputs
        texts = [ex["title"] for ex in batch]
        text_inputs = text_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )
        
        # Process images
        images = []
        for ex in batch:
            try:
                path = ex.get("image_path", "")
                if os.path.exists(path):
                    img = Image.open(path).convert("RGB")
                else:
                    # If image doesn't exist, use a blank white image
                    img = Image.new("RGB", (256, 256), (255, 255, 255))
            except Exception as e:
                print(f"Error loading image: {e}")
                img = Image.new("RGB", (256, 256), (255, 255, 255))
            images.append(img)
            
        # Process images with CLIP processor
        image_inputs = clip_processor(images=images, return_tensors="pt")
        
        # Get labels
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
        
        return {
            "input_ids": text_inputs["input_ids"].to(device),
            "attention_mask": text_inputs["attention_mask"].to(device),
            "pixel_values": image_inputs["pixel_values"].to(device),
            "labels": labels.to(device)
        }

    train_ds, val_ds, test_ds = stratified_split(ds, label_col="label", seed=args.seed)
    print("Sizes:", len(train_ds), len(val_ds), len(test_ds))

    # ... rest of your code remains the same ...
# 2) Tokenizers and processors
# -------------------------
print("Loading tokenizers / processors...")
# DeBERTa uses SentencePiece -> force slow tokenizer
# text_tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=False)
# text_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
text_tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
clip_processor = AutoProcessor.from_pretrained(args.image_model)
clip_model = SiglipModel.from_pretrained(args.image_model).to(device)  # used for image features

# Optionally load text model separately (we'll load inside the model class)
text_model_name = args.text_model
image_model_name = args.image_model

import os
import aiohttp
import asyncio
from aiohttp import ClientTimeout
from io import BytesIO
from PIL import Image
from datasets import load_dataset

SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# Async fetch function
# -------------------------
async def fetch_image(session, url, idx):
    save_path = os.path.join(SAVE_DIR, f"{idx}.jpg")
    if os.path.exists(save_path):
        return  # skip if already downloaded

    try:
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                content = await resp.read()
                img = Image.open(BytesIO(content)).convert("RGB")
                img.save(save_path, "JPEG")
            else:
                Image.new("RGB", (224, 224), (255, 255, 255)).save(save_path, "JPEG")
    except:
        Image.new("RGB", (224, 224), (255, 255, 255)).save(save_path, "JPEG")

# -------------------------
# Process one batch
# -------------------------
async def process_batch(urls, start_idx=0):
    timeout = ClientTimeout(total=10)
    connector = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        for i, url in enumerate(urls):
            idx = start_idx + i
            tasks.append(fetch_image(session, url, idx))
        await asyncio.gather(*tasks)

# -------------------------
# Download in 10k chunks
# -------------------------
def download_in_chunks(ds, chunk_size=10000):
    urls = ds["src"]  # column name must match your dataset
    for start in range(0, len(urls), chunk_size):
        end = min(start + chunk_size, len(urls))
        print(f"Downloading {start} → {end} ...")
        asyncio.run(process_batch(urls[start:end], start_idx=start))

# -------------------------
# Preprocessing (use local files)
# -------------------------
def preprocess_example(example, idx):
    # Text encoding
    enc = text_tokenizer(
        example["title"],
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_attention_mask=True,
    )

    # Store image path only; load/transform in collate_fn to avoid huge Arrow arrays
    img_path = os.path.join(SAVE_DIR, f"{idx}.jpg")
    example["input_ids"] = enc["input_ids"]
    example["attention_mask"] = enc["attention_mask"]
    example["image_path"] = img_path
    return example

# -------------------------
# Chunked processing helpers (to cap disk usage)
# -------------------------

def iter_chunk_ranges(n, chunk_size):
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield start, end

def download_range(urls, offset, network_batch=10000):
    for start in range(0, len(urls), network_batch):
        end = min(start + network_batch, len(urls))
        print(f"Downloading {offset + start} → {offset + end} ...")
        asyncio.run(process_batch(urls[start:end], start_idx=offset + start))

def preprocess_example_with_offset(example, idx, offset):
    return preprocess_example(example, idx + offset)

# Get number of unique labels from the dataset
num_labels = len(set(ds["label"]))
print(f"Found {num_labels} unique classes in the dataset")

# Initialize model
print("Initializing model...")
# Initialize model with text finetuning always enabled and image encoder frozen initially
model = MultimodalClassifier(
    text_model_name=args.text_model,
    clip_model=clip_model,
    num_labels=num_labels,
    text_finetune=True,  # Always enable text finetuning
    clip_finetune=False  # Start with frozen image encoder
).to(device)

# Ensure text encoder is trainable
for param in model.text_encoder.parameters():
    param.requires_grad = True

# Set up optimizer
print("Setting up optimizer...")
# Use a single learning rate for all parameters, matching V1's approach
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
print("Using single learning rate: 2e-5 for all parameters")

# Set up learning rate scheduler
total_steps = (len(train_ds) // args.batch_size) * args.epochs
warmup_steps = int(0.1 * total_steps)  # 10% of training steps for warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Set gradient accumulation to 1 to match V1
args.gradient_accumulation_steps = 1

print(f"Total training steps: {total_steps}")
print(f"Warmup steps: {warmup_steps}")
print(f"Cosine annealing steps: {total_steps - warmup_steps}")

def evaluate_over_dataset(dataset, name="val"):
    """Evaluate the model on the given dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating on {name}"):
            # Forward pass
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"]
            )
            
            # Get predictions
            logits = out["logits"]
            preds = torch.argmax(logits, dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            total_loss += out["loss"].item() * len(batch["labels"])
    
    # Calculate metrics
    avg_loss = total_loss / len(dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"\n{name} Evaluation:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro-F1: {f1:.4f}")
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": f1,
        "preds": all_preds,
        "labels": all_labels
    }

# Training loop
# Initialize training state
best_val_f1 = 0.0
total_steps_done = 0
last_checkpoint_time = time.time()
    
# Create checkpoint directory if it doesn't exist
os.makedirs(args.checkpoint_dir, exist_ok=True)

def unfreeze_image_encoder_layers(model, num_layers):
    """Unfreeze the last 'num_layers' of the image encoder."""
    if num_layers == 0:
        # Freeze all layers
        for param in model.clip_model.parameters():
            param.requires_grad = False
    else:
        # First freeze all layers
        for param in model.clip_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last 'num_layers' transformer blocks in the vision encoder
        vision_encoder = model.clip_model.vision_model.encoder
        total_layers = len(vision_encoder.layers)
        
        # Unfreeze the last 'num_layers' transformer blocks
        for i in range(max(0, total_layers - num_layers), total_layers):
            for param in vision_encoder.layers[i].parameters():
                param.requires_grad = True
            
        # Also unfreeze the projection layer
        if hasattr(model.clip_model, 'visual_projection'):
            for param in model.clip_model.visual_projection.parameters():
                param.requires_grad = True

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    step_count = 0
    
    # Set up progressive unfreezing based on epoch
    if epoch == 0:
        # Epoch 1: Freeze image encoder completely
        unfreeze_image_encoder_layers(model, 0)
    elif epoch == 1:
        # Epoch 2: Unfreeze last 1 layer of image encoder
        unfreeze_image_encoder_layers(model, 1)
    else:
        # Epoch 3+: Unfreeze last 2 layers of image encoder
        unfreeze_image_encoder_layers(model, 2)
    
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{args.epochs}")
    print(f"Unfrozen image encoder layers: {model.clip_model.vision_model.encoder.layers[-2:] if epoch > 0 else 'None'}")
    print(f"{'='*50}")

    # Process dataset in chunks to manage memory
    for start, end in iter_chunk_ranges(len(train_ds), args.chunk_size):
        # 1) Download images for this chunk
        urls_chunk = train_ds["src"][start:end]
        download_range(urls_chunk, offset=start, network_batch=10000)

        # 2) Preprocess this chunk
        train_chunk = train_ds.select(range(start, end))
        train_chunk = train_chunk.map(
            lambda ex, idx: preprocess_example_with_offset(ex, idx, start), 
            with_indices=True
        )

        # 3) Create data loader for this chunk
        loader = DataLoader(
            train_chunk, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )

        # 4) Training loop for this chunk
        pbar = tqdm(loader, desc=f"Train {start}-{end}")
        for step, batch in enumerate(pbar):
            # Forward pass
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"]
            )
            loss = out["loss"]
            
            # Backward pass with gradient accumulation
            (loss / args.gradient_accumulation_steps).backward()
            
            # Only step the optimizer after accumulating enough gradients
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(loader) - 1:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update model parameters
                optimizer.step()
                optimizer.zero_grad()
                
                # Update learning rate if using a scheduler
                if scheduler is not None:
                    scheduler.step()
            
            # Update training metrics
            running_loss += loss.item()
            step_count += 1
            total_steps_done += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{running_loss/step_count:.4f}"})
            
        # Save checkpoint periodically
        current_time = time.time()
        if current_time - last_checkpoint_time >= args.checkpoint_interval:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f"checkpoint_epoch_{epoch+1}_step_{total_steps_done}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'best_val_f1': best_val_f1,
                'total_steps_done': total_steps_done,
                'args': vars(args)
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
            last_checkpoint_time = current_time

        # Cleanup downloaded images for this chunk
        cleanup_images_range(start, end)

        # Validate over val set in chunks (after all training chunks are processed)
        val_metrics = evaluate_over_dataset(val_ds, name="val")
        print(f"\nEpoch {epoch+1} Val Loss: {val_metrics['loss']:.4f} Macro-F1: {val_metrics['macro_f1']:.4f} Acc: {val_metrics['accuracy']:.4f}")

        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_model_path = os.path.join(args.checkpoint_dir, "best_multimodal.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path} with F1: {best_val_f1:.4f}")
            
            # Also save a full checkpoint with optimizer state
            checkpoint_path = os.path.join(args.checkpoint_dir, f"best_checkpoint_epoch_{epoch+1}_f1_{best_val_f1:.4f}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'best_val_f1': best_val_f1,
                'total_steps_done': total_steps_done,
                'args': vars(args),
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"Saved full best checkpoint to {checkpoint_path}")

    # Final test evaluation (after all epochs)
    print("=== Test evaluation on best model ===")
    model.load_state_dict(torch.load("checkpoints/best_multimodal.pt"))
    test_metrics = evaluate_over_dataset(test_ds, name="test")
    print(f"Test Loss: {test_metrics['loss']:.4f} Macro-F1: {test_metrics['macro_f1']:.4f} Acc: {test_metrics['accuracy']:.4f}")
    print("Classification report:")
    print(classification_report(test_metrics["labels"], test_metrics["preds"], target_names=labels, zero_division=0))
