#!/usr/bin/env python3
"""
Model Evaluation Script
Loads the trained best_multimodal.pt model and evaluates its accuracy on test data.
"""

import argparse
import os
import random
import asyncio
import aiohttp
from io import BytesIO

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    SiglipModel,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report
from model_arch import MultimodalClassifier

# -------------------------
# Config / Arguments
# -------------------------
parser = argparse.ArgumentParser(description="Evaluate trained multimodal model")
# parser.add_argument("--csv", type=str, default="all_products.csv", help="Merged CSV path")
parser.add_argument("--csv", type=str, default="all_products50.csv", help="Merged CSV path")
parser.add_argument("--text_model", type=str, default="microsoft/deberta-v3-small",
                   help="HuggingFace model name for text encoder")
parser.add_argument("--image_model", type=str, default="google/siglip2-base-patch16-256",
                   help="HuggingFace model name for image encoder")
parser.add_argument("--max_length", type=int, default=64,
                   help="Maximum sequence length for text input")
parser.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for evaluation")
parser.add_argument("--sample", type=int, default=0,
                   help="If >0, sample this many rows for quick evaluation")
parser.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
parser.add_argument("--chunk_size", type=int, default=700000,
                   help="Process dataset in chunks to limit disk usage")
parser.add_argument("--model_path", type=str, default="checkpoints/checkpoint_epoch_3_step_522333.pt",
                   help="Path to trained model checkpoint")
parser.add_argument("--hidden_dim", type=int, default=768,
                   help="Hidden dimension size for the model")
parser.add_argument("--num_heads", type=int, default=8,
                   help="Number of attention heads in transformer layers")
parser.add_argument("--num_transformer_layers", type=int, default=6,
                   help="Number of transformer layers for fusion")
parser.add_argument("--classifier_hidden", type=int, default=512,
                   help="Hidden size of the classifier head")
args = parser.parse_args()

# Reproducibility
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Loading model from: {args.model_path}")

# Check if model file exists
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model file not found: {args.model_path}")

# -------------------------
# 1) Load CSV and prepare data
# -------------------------
print(f"Loading CSV: {args.csv}")
ds = load_dataset("csv", data_files=args.csv, split="train")
ds = ds.filter(lambda x: x["classified_niche_ai"] is not None)

# If you want a small quick sample (useful for debugging)
if args.sample and args.sample > 0:
    print(f"Sampling {args.sample} rows for quick evaluation")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.sample, len(ds))))

# Ensure columns expected: title, src (image url), niche
expected = ["title", "src", "classified_niche_ai"]
for c in expected:
    if c not in ds.column_names:
        raise ValueError(f"CSV must contain column '{c}'. Found: {ds.column_names}")

# Encode labels (must match training)
current_labels = sorted(list(set(ds["classified_niche_ai"])))
print(f"Current dataset labels ({len(current_labels)}): {current_labels}")

# We'll determine the correct labels after checking the saved model
labels = current_labels  # temporary
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(labels)

print(f"Number of labels: {num_labels}")

def add_label(example):
    example["label"] = label2id[example["classified_niche_ai"]]
    return example

ds = ds.map(add_label)

# -------------------------
# 2) Create test split (same as training)
# -------------------------
print("Creating test split (stratified)...")
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
print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# -------------------------
# 3) Load tokenizers and processors
# -------------------------
print("Loading tokenizers / processors...")
text_tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
clip_processor = AutoProcessor.from_pretrained(args.image_model)
clip_model = SiglipModel.from_pretrained(args.image_model).to(device)

text_model_name = args.text_model
image_model_name = args.image_model

# -------------------------
# 4) Image download functions (same as training)
# -------------------------
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

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

async def process_batch(urls, start_idx=0):
    timeout = aiohttp.ClientTimeout(total=10)
    connector = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        for i, url in enumerate(urls):
            idx = start_idx + i
            tasks.append(fetch_image(session, url, idx))
        await asyncio.gather(*tasks)

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
# 5) DataLoader collate_fn
# -------------------------
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

# -------------------------
# 7) Load trained model
# -------------------------
print("Creating model architecture...")

# First, let's check what's in the saved model to determine the correct number of classes
print("Checking saved model structure...")
try:
    saved_state = torch.load(args.model_path, map_location='cpu')
    
    # Try to determine number of classes from the saved state
    num_labels = None
    
    # Check for classifier weights in the saved state
    for k in saved_state.keys():
        if k.endswith('weight') and 'classifier' in k and len(saved_state[k].shape) == 2:
            if saved_state[k].shape[0] > 10:  # Assuming classifier output is at least 10 classes
                num_labels = saved_state[k].shape[0]
                print(f"Found classifier with {num_labels} classes in saved model")
                break
    
    if num_labels is None:
        print("⚠️  Could not determine number of classes from saved model, using current dataset classes")
        num_labels = len(current_labels)
    
    print(f"Using {num_labels} classes")
    
    # Initialize model with the correct number of classes and architecture params
    model = MultimodalClassifier(
        text_model_name=args.text_model,
        clip_model=clip_model,
        num_labels=num_labels,
        hidden_dim=args.hidden_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        classifier_hidden=args.classifier_hidden
    ).to(device)
    
    print(f"Model architecture:")
    print(f"- Hidden dim: {args.hidden_dim}")
    print(f"- Transformer layers: {args.num_transformer_layers}")
    print(f"- Attention heads: {args.num_heads}")
    print(f"- Classifier hidden: {args.classifier_hidden}")
    print(f"- Number of classes: {num_labels}")
    
    # Load the state dict with strict=False to handle architecture changes
    print(f"Loading trained weights from {args.model_path}...")
    try:
        model.load_state_dict(saved_state, strict=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: {str(e)}")
        print("⚠️  Some weights could not be loaded. This is normal if you've changed the architecture.")
        
        # Try partial loading
        model_dict = model.state_dict()
        # 1. Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in saved_state.items() if k in model_dict}
        # 2. Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. Load the new state dict
        model.load_state_dict(model_dict, strict=False)
        print("✅ Loaded partial weights successfully!")
        
    model.eval()
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("Failed to load model. Please check the model file and architecture.")
    exit(1)

# -------------------------
# 8) Evaluation function
# -------------------------
def evaluate_over_dataset(dataset, name="test"):
    print(f"\nEvaluating on {name} dataset...")
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for start, end in iter_chunk_ranges(len(dataset), args.chunk_size):
            print(f"Processing chunk {start}-{end} of {len(dataset)}")
            
            # download images for this chunk
            urls_chunk = dataset["src"][start:end]
            download_range(urls_chunk, offset=start, network_batch=10000)

            # preprocess this chunk (store paths)
            chunk = dataset.select(range(start, end))
            chunk = chunk.map(lambda ex, idx: preprocess_example_with_offset(ex, idx, start), with_indices=True)

            loader = DataLoader(chunk, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            
            for batch in tqdm(loader, desc=f"Evaluating {start}-{end}"):
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

# -------------------------
# 9) Run evaluation
# -------------------------
print("=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

test_metrics = evaluate_over_dataset(test_ds, name="test")

print(f"\n📊 TEST RESULTS:")
print(f"   Test Loss: {test_metrics['loss']:.4f}")
print(f"   Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
print(f"   Test Macro-F1: {test_metrics['macro_f1']:.4f}")

print(f"\n📈 DETAILED CLASSIFICATION REPORT:")
# Handle the case where we have more classes in the model than in current labels
unique_labels = sorted(list(set(test_metrics["labels"] + test_metrics["preds"])))
if len(unique_labels) <= len(labels):
    # Use current labels if they cover all predictions
    target_names = labels
else:
    # Create generic names for missing classes
    target_names = [f"Class_{i}" for i in range(max(unique_labels) + 1)]
    print(f"Note: Using generic class names due to label mismatch")

print(classification_report(test_metrics["labels"], test_metrics["preds"], 
                          target_names=target_names, zero_division=0))

# Save results to file
results_file = "evaluation_results.txt"
with open(results_file, "w") as f:
    f.write("MODEL EVALUATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model Path: {args.model_path}\n")
    f.write(f"Dataset: {args.csv}\n")
    f.write(f"Test Size: {len(test_ds)}\n")
    f.write(f"Number of Labels: {num_labels}\n\n")
    f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
    f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)\n")
    f.write(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(test_metrics["labels"], test_metrics["preds"], target_names=labels, zero_division=0))

print(f"\n💾 Results saved to: {results_file}")
print("=" * 60)
