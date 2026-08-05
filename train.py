import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
import dataset
import models


def _step_loss_and_acc(pitch_logits, dur_logits, y_pitch, y_dur, criterion):
    loss = 0.0
    pitch_correct, dur_correct, total = 0, 0, 0
    for v in range(config.NUM_VOICES):
        loss = loss + criterion(pitch_logits[v], y_pitch[:, v])
        loss = loss + criterion(dur_logits[v], y_dur[:, v])
        pitch_correct += (pitch_logits[v].argmax(-1) == y_pitch[:, v]).sum().item()
        dur_correct += (dur_logits[v].argmax(-1) == y_dur[:, v]).sum().item()
        total += y_pitch.size(0)
    return loss, pitch_correct / total, dur_correct / total


def _run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, pitch_acc_sum, dur_acc_sum, n_batches = 0.0, 0.0, 0.0, 0

    for xp, xd, yp, yd in loader:
        xp, xd, yp, yd = xp.to(config.DEVICE), xd.to(config.DEVICE), yp.to(config.DEVICE), yd.to(config.DEVICE)

        if is_train:
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=config.USE_AMP):
                pitch_logits, dur_logits, _, _ = model(xp, xd, yp, yd)
                loss, p_acc, d_acc = _step_loss_and_acc(pitch_logits, dur_logits, yp, yd, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                pitch_logits, dur_logits, _, _ = model(xp, xd, yp, yd)
                loss, p_acc, d_acc = _step_loss_and_acc(pitch_logits, dur_logits, yp, yd, criterion)


        total_loss += loss.item()
        pitch_acc_sum += p_acc
        dur_acc_sum += d_acc
        n_batches += 1

    return total_loss / n_batches, pitch_acc_sum / n_batches, dur_acc_sum / n_batches


def _safe_torch_save(obj, f_path, max_retries=5):
    for i in range(max_retries):
        try:
            tmp_p = f_path + ".tmp"
            torch.save(obj, tmp_p)
            if os.path.exists(f_path):
                os.remove(f_path)
            os.replace(tmp_p, f_path)
            return
        except Exception:
            if i == max_retries - 1:
                torch.save(obj, f_path)
            else:
                time.sleep(0.5)


def train_model(model_name, bundle):
    config.set_seed()
    pitch_vocab, dur_vocab = bundle["pitch_vocab"], bundle["dur_vocab"]

    train_ds = dataset.ChoraleWindowDataset(bundle["train_pieces"], pitch_vocab, dur_vocab)
    val_ds = dataset.ChoraleWindowDataset(bundle["val_pieces"], pitch_vocab, dur_vocab)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = models.build_model(model_name, len(pitch_vocab), len(dur_vocab)).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_pitch_acc": [], "train_dur_acc": [],
               "val_pitch_acc": [], "val_dur_acc": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    ckpt_path = os.path.join(config.CKPT_DIR, f"{model_name}_best.pt")

    for epoch in range(config.MAX_EPOCHS):
        t0 = time.time()
        train_loss, train_p_acc, train_d_acc = _run_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_p_acc, val_d_acc = _run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_pitch_acc"].append(train_p_acc)
        history["train_dur_acc"].append(train_d_acc)
        history["val_pitch_acc"].append(val_p_acc)
        history["val_dur_acc"].append(val_d_acc)

        print(f"[{model_name}] epoch {epoch+1} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_pitch_acc={val_p_acc:.3f} val_dur_acc={val_d_acc:.3f} "
              f"({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            _safe_torch_save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                print(f"[{model_name}] early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    with open(os.path.join(config.OUTPUT_DIR, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f)

    return model, history



def evaluate_test_set(model, bundle):
    pitch_vocab, dur_vocab = bundle["pitch_vocab"], bundle["dur_vocab"]
    test_ds = dataset.ChoraleWindowDataset(bundle["test_pieces"], pitch_vocab, dur_vocab)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_p_acc, test_d_acc = _run_epoch(model, test_loader, criterion)
    return {"test_loss": test_loss, "test_pitch_acc": test_p_acc, "test_dur_acc": test_d_acc}


if __name__ == "__main__":
    bundle = dataset.prepare_data()
    results = {}
    for name in config.MODEL_NAMES:
        model, history = train_model(name, bundle)
        results[name] = evaluate_test_set(model, bundle)
        print(name, results[name])
    with open(os.path.join(config.OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
