#!/usr/bin/env python3

import time
import wandb
import torch
import numpy as np
import random
import sys

from config import (SEED, NUM_CLIENTS, FRACTION, NUM_ROUNDS, LOCAL_EPOCHS,
                    BATCH_SIZE, FL_LR, PRETRAIN_EPOCHS, PRETRAIN_LR, NUM_CLASSES,
                    DATA_PER_CLIENT, INV_RESTARTS)
from fl_training import run_fl_training
from class_unlearning import retrain_without_classes, fuia_multiclass_unlearning_attack


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_user_target_classes():
    print("=" * 60)
    print(" CONFIGURAZIONE MULTI-CLASS UNLEARNING")
    print("=" * 60)
    
    # 1. Scelta della quantità di classi
    while True:
        try:
            k_input = input("Quante classi vuoi rimuovere? (Scegli tra 2 e 4): ").strip()
            num_classes_to_remove = int(k_input)
            if 2 <= num_classes_to_remove <= 4:
                break
            print(">> Errore: inserisci un numero compreso tra 2 e 4.")
        except ValueError:
            print(">> Errore: input non valido.")

    # 2. Scelta delle classi specifiche
    while True:
        try:
            raw_input = input(f"Inserisci le {num_classes_to_remove} classi separate da spazio (es. 3 7): ").strip()
            classes = [int(c) for c in raw_input.replace(",", " ").split()]
            classes = list(set(classes))  # Rimuove eventuali duplicati
            
            if len(classes) == num_classes_to_remove and all(0 <= c <= 9 for c in classes):
                return sorted(classes)
            print(f">> Errore: Devi inserire esattamente {num_classes_to_remove} cifre distinte tra 0 e 9.")
        except ValueError:
            print(">> Errore: formato non valido. Inserisci solo numeri separati da spazio.")


if __name__ == "__main__":
    target_classes = get_user_target_classes()
    k_targets = len(target_classes)
    beta_param = 0.5

    set_seed(SEED)

    wandb.init(project="FUIA_Class_Unlearning", config={
        "scenario": "multi_class_unlearning",
        "dataset": "MNIST",
        "unlearning_method": "retraining",
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "seed": SEED,
        "target_classes": target_classes,
        "num_unlearned_classes": k_targets
    })

    total_t0 = time.time()

    # Fase 1: Federated Learning Completo
    print("\n" + "-" * 60)
    print("Fase 1: Federated Learning (Modello Originale W^o)")
    print("-" * 60)
    (original_model, stored_updates, client_data, private_data,
     pretrained_sd, round_selections, test_loader) = run_fl_training()

    # Reset esplicito del seed prima del Retraining
    set_seed(SEED)

    # Fase 2: Multi-Class Unlearning (Retraining senza le classi target)
    print("\n" + "-" * 60)
    print(f"Fase 2: Multi-Class Unlearning (Retraining senza classi: {target_classes})")
    print("-" * 60)
    unlearned_model = retrain_without_classes(
        pretrained_sd, private_data, client_data, target_classes, round_selections, test_loader
    )

    # Fase 3: Attacco FUIA Multi-Class Unlearning (Top-k)
    print("\n" + "-" * 60)
    print(f"Fase 3: Attacco FUIA Multi-Class Unlearning (Top-{k_targets} Score S_d)")
    print("-" * 60)
    predicted_classes, scores = fuia_multiclass_unlearning_attack(
        original_model, unlearned_model, top_k=k_targets, beta=beta_param
    )

    print("\n" + "=" * 60)
    print(f" RISULTATI ATTACCO MULTI-CLASS UNLEARNING (N_uc = {k_targets})")
    print("=" * 60)
    for c_id, score in enumerate(scores):
        mark = "<-- TOP CLASSE PREDETTA" if c_id in predicted_classes else ""
        print(f"  Classe {c_id}: Score S_d = {score:.4f} {mark}")

    # Conteggio classi correttamente identificate (N_ic)
    correct_inferred = set(target_classes).intersection(set(predicted_classes))
    n_ic = len(correct_inferred)
    is_full_success = (n_ic == k_targets)

    print("-" * 60)
    print(f"  Classi Rimosse Reali (N_uc):        {target_classes}")
    print(f"  Classi Predette dall'Attacco:       {predicted_classes}")
    print(f"  Classi Corrette Identificate (N_ic): {n_ic} su {k_targets}")
    print(f"  Esito Attacco:                      {'FULL SUCCESS (100%)' if is_full_success else f'PARTIAL ({n_ic}/{k_targets})'}")
    print("=" * 60)

    total_time = time.time() - total_t0
    print(f"\nTempo totale: {total_time:.0f}s ({total_time / 60:.1f} min)")
    wandb.log({
        "attack/full_success": is_full_success,
        "attack/n_ic": n_ic,
        "attack/n_uc": k_targets,
        "total_time_s": total_time
    })
    wandb.finish()