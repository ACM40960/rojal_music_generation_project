"""
Evaluation Script for Bach Chorale Completion.
Computes:
1. Prediction Metrics (1-Step Ahead on Validation Set): Pitch Accuracy, Duration MSE (beats), Time-to-Next MSE (beats).
2. Generation Metrics (Autoregressive Rollout on Target Song): Pitch Class Entropy, Scale Consistency, Pitch Range, Unique Pitches.
"""

import os
import pickle
import argparse
import numpy as np
import muspy

# Monkey-patch pyparsing for TensorFlow/httplib2 compatibility
import pyparsing
if not hasattr(pyparsing, 'DelimitedList'):
    pyparsing.DelimitedList = pyparsing.delimitedList

from tensorflow import keras
from sklearn.model_selection import train_test_split

# Import local modules
from data_pipeline import tensor_to_muspy
from models import build_recurrent_model, LinearAutoregressiveModel
from train import prepare_recurrent_dataset

def generate_autoregressive(model, model_type, seed_seq, steps_to_gen, seq_len=16):
    """
    Generates continuation autoregressively.
    seed_seq: shape (K, 4, 3)
    """
    generated = []
    current_history = np.copy(seed_seq)
    
    for k in range(steps_to_gen):
        if model_type == 'Linear':
            next_step = model.predict(current_history[-model.lag:])
        else:
            input_seq = current_history[-seq_len:]
            input_seq_flat = input_seq.reshape(1, seq_len, 12)
            preds = model.predict(input_seq_flat, verbose=0)
            
            next_step = np.zeros((4, 3))
            for v in range(4):
                next_step[v, 0] = np.argmax(preds[v][0])
            next_step[:, 1] = preds[4][0]
            next_step[:, 2] = preds[5][0]
            
        generated.append(next_step)
        current_history = np.vstack([current_history, next_step[np.newaxis, ...]])
        
    return np.array(generated)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bach Chorale Completion")
    parser.add_argument('--dataset', type=str, default='processed_dataset.pkl', help='Path to dataset file')
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"[ERROR] Dataset {args.dataset} not found. Run data_pipeline.py first.")
        
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)
        
    print(f"[INFO] Evaluating models on dataset ({len(dataset)} chorales)...")
    
    # 1. Prepare validation set for 1-step prediction metrics
    X, y = prepare_recurrent_dataset(dataset, seq_len=16)
    indices = np.arange(len(X))
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_val = X[val_idx]
    y_val = {key: val[val_idx] for key, val in y.items()}
    
    # 2. Target song for multi-step autoregressive generation
    test_song = dataset[0]['tensor']
    bwv = dataset[0]['bwv']
    cut_point = len(test_song) // 2
    seed_seq = test_song[:cut_point]
    true_continuation = test_song[cut_point:]
    steps_to_gen = len(true_continuation)
    
    models = {}
    
    # Load Linear baseline
    linear_path = 'saved_models/linear_baseline.pkl'
    if os.path.exists(linear_path):
        with open(linear_path, 'rb') as f:
            models['Linear'] = pickle.load(f)
            
    # Load Keras models
    for m_type in ['rnn', 'lstm', 'gru']:
        keras_path = f'saved_models/model_{m_type}.keras'
        if os.path.exists(keras_path):
            models[m_type.upper()] = keras.models.load_model(keras_path)
            
    if not models:
        print("[ERROR] No models found in saved_models/. Run train.py first.")
        return
        
    print("\n" + "="*85)
    print("                      1-STEP AHEAD PREDICTION METRICS (VALIDATION SET)")
    print("="*85)
    print(f"{'Model':<12} | {'Val Pitch Acc':<15} | {'Val Duration MSE (beats)':<25} | {'Val T2N MSE (beats)':<25}")
    print("-"*85)
    
    val_metrics = {}
    
    for name, model in models.items():
        if name == 'Linear':
            # Linear 1-step evaluation
            X_val_seq = X_val.reshape(len(X_val), 16, 12)[:, -4:, :] # lag=4
            preds_linear = np.array([model.predict(seq) for seq in X_val_seq])
            
            pitch_acc = np.mean([
                np.mean(np.round(preds_linear[:, v, 0]) == y_val[f'pitch_{v}']) for v in range(4)
            ])
            dur_mse = np.mean((preds_linear[:, :, 1] - y_val['duration'])**2)
            t2n_mse = np.mean((preds_linear[:, :, 2] - y_val['time_to_next'])**2)
        else:
            preds = model.predict(X_val, verbose=0)
            
            pitch_acc = np.mean([
                np.mean(np.argmax(preds[v], axis=1) == y_val[f'pitch_{v}']) for v in range(4)
            ])
            dur_mse = np.mean((preds[4] - y_val['duration'])**2)
            t2n_mse = np.mean((preds[5] - y_val['time_to_next'])**2)
            
        val_metrics[name] = {'pitch_acc': pitch_acc, 'dur_mse': dur_mse, 't2n_mse': t2n_mse}
        print(f"{name:<12} | {pitch_acc:<15.4f} | {dur_mse:<25.4f} | {t2n_mse:<25.4f}")
        
    print("="*85 + "\n")
    
    print("="*85)
    print("                AUTOREGRESSIVE GENERATION METRICS (BWV " + str(bwv) + ")")
    print("="*85)
    print(f"{'Model':<12} | {'Pitch Class Entropy':<20} | {'Scale Consistency':<18} | {'Pitch Range':<12} | {'Unique Pitches':<12}")
    print("-"*85)
    
    for name, model in models.items():
        gen_continuation = generate_autoregressive(model, name, seed_seq, steps_to_gen)
        music_obj = tensor_to_muspy(gen_continuation)
        
        try:
            pce = muspy.metrics.pitch_class_entropy(music_obj)
            sc = muspy.metrics.scale_consistency(music_obj)
        except Exception:
            pce, sc = np.nan, np.nan
            
        flat_pitches = [note.pitch for track in music_obj.tracks for note in track.notes]
        pitch_range = (max(flat_pitches) - min(flat_pitches)) if flat_pitches else 0
        unique_pitches = len(set(flat_pitches)) if flat_pitches else 0
        
        pce_str = f"{pce:.4f}" if not np.isnan(pce) else "N/A"
        sc_str = f"{sc:.4f}" if not np.isnan(sc) else "N/A"
        
        print(f"{name:<12} | {pce_str:<20} | {sc_str:<18} | {pitch_range:<12} | {unique_pitches:<12}")
        
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
