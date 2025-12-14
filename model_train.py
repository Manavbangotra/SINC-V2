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
parser.add_argument("--epochs", type=int, default=5, 
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
        # batch is a list of examples
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
        # Load images on-the-fly to avoid storing large tensors in the dataset
        images = []
        for ex in batch:
            path = ex.get("image_path")
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (255, 255, 255))
            images.append(img)
        pixel_values = clip_processor(images=images, return_tensors="pt")["pixel_values"]  # (B,C,H,W)
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "pixel_values": pixel_values.to(device),
            "labels": labels.to(device),
        }

    # Instantiate model
    num_labels = len(labels)
    print(labels)
    print("Num labels:", num_labels)
    # Initialize model with frozen encoders by default
    model = MultimodalClassifier(
        text_model_name=args.text_model,
        clip_model=clip_model,
        num_labels=len(labels),
        hidden_dim=768,
        num_transformer_layers=4,  # Using 4 layers as recommended
        num_heads=8,
        classifier_hidden=512
    )
    
    # Freeze all parameters by default
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze classifier and fusion components
    for module in [model.classifier, model.transformer]:
        for param in module.parameters():
            param.requires_grad = True
            
    # Unfreeze individual parameters
    for param in [model.cls_token, model.text_mod_emb, model.img_mod_emb]:
        if hasattr(param, 'requires_grad'):
            param.requires_grad = True
    model.to(device)

    # -------------------------
    # 6) Parameter grouping and optimizer setup
    # -------------------------
    # Create parameter groups with different learning rates
    optimizer_groups = [
        # Classifier head (highest LR)
        {"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": args.lr_mlp},
        
        # Fusion components (medium LR)
        {"params": [p for p in model.transformer.parameters() if p.requires_grad], "lr": args.lr_projection},
        {"params": [model.cls_token, model.text_mod_emb, model.img_mod_emb], "lr": args.lr_projection},
    ]
    
    # Add text encoder parameters if unfrozen
    if hasattr(model, 'text_encoder') and args.unfreeze_text_layers > 0:
        text_encoder = model.text_encoder
        # Only get the last N layers to unfreeze
        layers_to_unfreeze = text_encoder.encoder.layer[-args.unfreeze_text_layers:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        optimizer_groups.append({
            "params": [p for p in layers_to_unfreeze.parameters() if p.requires_grad],
            "lr": args.lr_text,
            "weight_decay": 0.01
        })
    
    # Add image encoder parameters if unfrozen
    if hasattr(model, 'clip_model') and args.unfreeze_image_layers > 0:
        clip_model = model.clip_model
        # SigLIP typically has a vision model with layers in .encoder.layers
        if hasattr(clip_model, 'vision_model') and hasattr(clip_model.vision_model, 'encoder'):
            vision_encoder = clip_model.vision_model.encoder
            # Calculate how many layers to unfreeze (last N%)
            total_layers = len(vision_encoder.layers)
            layers_to_unfreeze = max(1, int(total_layers * (args.unfreeze_image_layers / 100.0)))
            for layer in vision_encoder.layers[-layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
            optimizer_groups.append({
                "params": [p for p in vision_encoder.layers[-layers_to_unfreeze:].parameters() 
                          if p.requires_grad],
                "lr": args.lr_image,
                "weight_decay": 0.01
            })
    
    # Create optimizer with parameter groups
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)
    
    def compute_total_steps_across_chunks(dataset_len, batch_size, chunk_size, epochs):
        if dataset_len == 0:
            return 0
        steps_per_epoch = 0
        for start, end in iter_chunk_ranges(dataset_len, chunk_size):
            chunk_len = end - start
            steps_per_epoch += math.ceil(chunk_len / batch_size)
        return steps_per_epoch * epochs
    
    # Calculate effective batch size and total steps
    effective_batch_size = args.batch_size * (args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 1)
    total_steps = compute_total_steps_across_chunks(
        len(train_ds), 
        effective_batch_size,  # Use effective batch size for step calculation
        args.chunk_size, 
        args.epochs
    )
    
    # Cosine learning rate scheduler with warmup
    def get_lr_lambda(current_step: int, num_warmup_steps: int, num_training_steps: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    # Calculate warmup steps (10% of total training steps)
    num_warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda step: get_lr_lambda(step, num_warmup_steps, total_steps)
            for _ in range(len(optimizer.param_groups))
        ]
    )
    
    # More aggressive warmup (10% of training)
    num_warmup = int(0.1 * total_steps)
    
    # Log training configuration
    print(f"\nTraining configuration:")
    print(f"- Batch size: {args.batch_size} (per device)")
    if hasattr(args, 'gradient_accumulation_steps'):
        print(f"- Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"- Effective batch size: {effective_batch_size}")
    print(f"- Total training steps: {total_steps}")
    print(f"- Warmup steps: {num_warmup} ({num_warmup/total_steps*100:.1f}%)")
    print(f"- Learning rates: MLP={args.lr_mlp}, Projection={args.lr_projection}, "
          f"Text={args.lr_text}, Image={args.lr_image}")
    
    # Create scheduler with linear warmup and cosine decay
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup, 
        num_training_steps=total_steps
    )

    # -------------------------
    # 7) Training loop + eval
    # -------------------------
    def evaluate_over_dataset(dataset, name="val"):
        model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for start, end in iter_chunk_ranges(len(dataset), args.chunk_size):
                # download images for this chunk
                urls_chunk = dataset["src"][start:end]
                download_range(urls_chunk, offset=start, network_batch=10000)

                # preprocess this chunk (store paths)
                chunk = dataset.select(range(start, end))
                chunk = chunk.map(lambda ex, idx: preprocess_example_with_offset(ex, idx, start), with_indices=True)

                loader = DataLoader(chunk, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
                for batch in loader:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"]
                    )
                    loss = outputs["loss"]
                    logits = outputs["logits"]
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(batch["labels"].cpu().numpy().tolist())
                    total_loss += loss.item() if loss is not None else 0.0
                    total_batches += 1

                # cleanup chunk images
                cleanup_images_range(start, end)

        avg_loss = total_loss / max(1, total_batches)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        micro_acc = accuracy_score(all_labels, all_preds)
        return {"loss": avg_loss, "macro_f1": macro_f1, "accuracy": micro_acc, "preds": all_preds, "labels": all_labels}

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize training state
    start_epoch = 0
    best_val_f1 = 0.0
    total_steps_done = 0
    last_checkpoint_time = time.time()
    
    # Track gradient accumulation
    accumulation_steps = 0
    total_loss = 0.0

    # Load checkpoint if resuming
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Loading checkpoint from {args.resume_from}")
            checkpoint = torch.load(args.resume_from)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_f1 = checkpoint.get('best_val_f1', 0.0)
            total_steps_done = checkpoint.get('total_steps_done', 0)
            print(f"Resuming training from epoch {start_epoch + 1}, best_val_f1: {best_val_f1:.4f}")
        else:
            print(f"Warning: Checkpoint {args.resume_from} not found. Starting from scratch.")

    # Training loop
    # Progressive unfreezing schedule
    def update_unfreezing(epoch, model):
        # Update text encoder layers
        if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'encoder'):
            text_layers = model.text_encoder.encoder.layer
            # Unfreeze top N layers
            for i in range(len(text_layers) - 1, max(-1, len(text_layers) - args.unfreeze_text_layers - 1), -1):
                for param in text_layers[i].parameters():
                    param.requires_grad = True
            
            print(f"Epoch {epoch+1}: Unfroze top {min(args.unfreeze_text_layers, len(text_layers))} text encoder layers")
        
        # Update image encoder layers (assuming ViT architecture for SigLIP)
        if hasattr(model, 'clip_model') and hasattr(model.clip_model.vision_model, 'encoder'):
            img_layers = model.clip_model.vision_model.encoder.layers
            # Unfreeze top N layers
            for i in range(len(img_layers) - 1, max(-1, len(img_layers) - args.unfreeze_image_layers - 1), -1):
                for param in img_layers[i].parameters():
                    param.requires_grad = True
            
            print(f"Epoch {epoch+1}: Unfroze top {min(args.unfreeze_image_layers, len(img_layers))} image encoder layers")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        step_count = 0
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Update unfreezing based on epoch
        if epoch == 1:  # Start unfreezing in second epoch
            args.unfreeze_text_layers = 2
            args.unfreeze_image_layers = 2
            update_unfreezing(epoch, model)
        elif epoch == 3:  # Unfreeze more layers in later epochs if needed
            args.unfreeze_text_layers = 4
            args.unfreeze_image_layers = 4
            update_unfreezing(epoch, model)

        for start, end in iter_chunk_ranges(len(train_ds), args.chunk_size):
            # 1) download images for this train chunk
            urls_chunk = train_ds["src"][start:end]
            download_range(urls_chunk, offset=start, network_batch=10000)

            # 2) preprocess this chunk (stores paths)
            train_chunk = train_ds.select(range(start, end))
            train_chunk = train_chunk.map(lambda ex, idx: preprocess_example_with_offset(ex, idx, start), with_indices=True)

            loader = DataLoader(train_chunk, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

            pbar = tqdm(loader, desc=f"Train {start}-{end}")
            for step, batch in enumerate(pbar):
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"]
                )
                loss = out["loss"]
                # Backward pass with gradient accumulation
                (loss / args.gradient_accumulation_steps).backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(loader) - 1:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    optimizer.step()
                    
                    # Update learning rate
                    if scheduler is not None:
                        scheduler.step()
                        
                    # Zero gradients
                    optimizer.zero_grad()
                
                running_loss += loss.item()
                step_count += 1
                total_steps_done += 1
                
                # Update progress bar
                if step_count % 10 == 0:
                    pbar.set_postfix({"loss": f"{running_loss / max(1, step_count):.4f}"})
                
                # Check if it's time to save a checkpoint
                current_time = time.time()
                if current_time - last_checkpoint_time >= args.checkpoint_interval:
                    checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}_step_{total_steps_done}.pt")
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

            # 4) cleanup downloaded images for this chunk
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
