import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import config


def set_seed(seed=config.SEED):
    """Ensure complete reproducibility across PyTorch, NumPy, and Python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics_and_loss(pitch_logits, dur_logits, pitch_target, dur_target, criterion):
    """
    Computes CrossEntropy loss and per-token/joint accuracies.
    pitch_logits: List of 4 tensors (B, num_pitches)
    dur_logits: List of 4 tensors (B, num_durations) or None
    pitch_target: (B, 4)
    dur_target: (B, 4)
    """
    B = pitch_target.shape[0]
    total_loss = 0.0
    
    # Pitch Loss and Accuracy
    correct_pitch_count = 0
    pitch_preds = []
    for v in range(4):
        p_loss = criterion(pitch_logits[v], pitch_target[:, v])
        total_loss = total_loss + p_loss
        pred_p = torch.argmax(pitch_logits[v], dim=-1)
        pitch_preds.append(pred_p)
        correct_pitch_count += (pred_p == pitch_target[:, v]).sum().item()
    
    pitch_acc = correct_pitch_count / (B * 4.0)

    # Duration Loss and Accuracy
    if dur_logits is not None:
        correct_dur_count = 0
        dur_preds = []
        for v in range(4):
            d_loss = criterion(dur_logits[v], dur_target[:, v])
            total_loss = total_loss + d_loss
            pred_d = torch.argmax(dur_logits[v], dim=-1)
            dur_preds.append(pred_d)
            correct_dur_count += (pred_d == dur_target[:, v]).sum().item()
        
        dur_acc = correct_dur_count / (B * 4.0)
        
        # Joint Event Accuracy (all 4 pitches and all 4 durations correct)
        pitch_preds_stacked = torch.stack(pitch_preds, dim=1) # (B, 4)
        dur_preds_stacked = torch.stack(dur_preds, dim=1)     # (B, 4)
        
        p_match = (pitch_preds_stacked == pitch_target).all(dim=1)
        d_match = (dur_preds_stacked == dur_target).all(dim=1)
        joint_acc = (p_match & d_match).float().mean().item()
    else:
        dur_acc = 0.0
        joint_acc = pitch_acc

    return total_loss, pitch_acc, dur_acc, joint_acc


def train_model(model, train_loader, val_loader, model_name="model", epochs=config.EPOCHS, device=config.DEVICE):
    """
    Unified training loop for all models.
    Uses AdamW, ReduceLROnPlateau, Early Stopping, Gradient Clipping, and Mixed Precision.
    """
    set_seed(config.SEED)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pt")

    history = {
        "train_loss": [], "val_loss": [],
        "val_pitch_acc": [], "val_dur_acc": [], "val_joint_acc": [],
        "train_time_sec": 0.0, "avg_inference_time_ms": 0.0
    }

    print(f"\n--- Starting Training: {model_name} on {device} ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_batches = len(train_loader)

        for batch in train_loader:
            p_in = batch["pitch_in"].to(device)
            d_in = batch["dur_in"].to(device)
            p_tgt = batch["pitch_target"].to(device)
            d_tgt = batch["dur_target"].to(device)

            optimizer.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                p_logits, d_logits = model(p_in, d_in)
                loss, _, _, _ = compute_metrics_and_loss(p_logits, d_logits, p_tgt, d_tgt, criterion)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / n_train_batches

        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        val_p_acc_sum = 0.0
        val_d_acc_sum = 0.0
        val_j_acc_sum = 0.0
        n_val_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                p_in = batch["pitch_in"].to(device)
                d_in = batch["dur_in"].to(device)
                p_tgt = batch["pitch_target"].to(device)
                d_tgt = batch["dur_target"].to(device)

                with autocast(enabled=device.type == "cuda"):
                    p_logits, d_logits = model(p_in, d_in)
                    loss, p_acc, d_acc, j_acc = compute_metrics_and_loss(p_logits, d_logits, p_tgt, d_tgt, criterion)

                val_loss_sum += loss.item()
                val_p_acc_sum += p_acc
                val_d_acc_sum += d_acc
                val_j_acc_sum += j_acc

        avg_val_loss = val_loss_sum / n_val_batches
        avg_val_p_acc = val_p_acc_sum / n_val_batches
        avg_val_d_acc = val_d_acc_sum / n_val_batches
        avg_val_j_acc = val_j_acc_sum / n_val_batches

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_pitch_acc"].append(avg_val_p_acc)
        history["val_dur_acc"].append(avg_val_d_acc)
        history["val_joint_acc"].append(avg_val_j_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Pitch Acc: {avg_val_p_acc:.4f} | Dur Acc: {avg_val_d_acc:.4f} | Joint Acc: {avg_val_j_acc:.4f}")

        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}.")
                break

    total_train_time = time.time() - start_time
    history["train_time_sec"] = total_train_time

    # Evaluate Average Inference Latency per batch
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    inf_times = []
    sample_batch = next(iter(val_loader))
    p_in = sample_batch["pitch_in"].to(device)
    d_in = sample_batch["dur_in"].to(device)
    with torch.no_grad():
        for _ in range(50):
            t0 = time.time()
            _ = model(p_in, d_in)
            inf_times.append((time.time() - t0) * 1000.0)  # ms
    history["avg_inference_time_ms"] = float(np.mean(inf_times))

    print(f"Training Complete for {model_name}. Total Time: {total_train_time:.2f}s | Best Val Loss: {best_val_loss:.4f}")
    return history


def evaluate_test_set(model, test_loader, model_name="model", device=config.DEVICE):
    """Evaluates the loaded best model on the unseen test set."""
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    test_loss_sum = 0.0
    p_acc_sum = 0.0
    d_acc_sum = 0.0
    j_acc_sum = 0.0
    n_batches = len(test_loader)

    with torch.no_grad():
        for batch in test_loader:
            p_in = batch["pitch_in"].to(device)
            d_in = batch["dur_in"].to(device)
            p_tgt = batch["pitch_target"].to(device)
            d_tgt = batch["dur_target"].to(device)

            p_logits, d_logits = model(p_in, d_in)
            loss, p_acc, d_acc, j_acc = compute_metrics_and_loss(p_logits, d_logits, p_tgt, d_tgt, criterion)

            test_loss_sum += loss.item()
            p_acc_sum += p_acc
            d_acc_sum += d_acc
            j_acc_sum += j_acc

    results = {
        "model_name": model_name,
        "test_loss": test_loss_sum / n_batches,
        "pitch_acc": p_acc_sum / n_batches,
        "dur_acc": d_acc_sum / n_batches,
        "joint_acc": j_acc_sum / n_batches
    }
    print(f"\n=== Test Set Results [{model_name}] ===")
    print(f"Test Loss: {results['test_loss']:.4f} | Pitch Acc: {results['pitch_acc']:.4f} | "
          f"Dur Acc: {results['dur_acc']:.4f} | Joint Acc: {results['joint_acc']:.4f}")
    return results
