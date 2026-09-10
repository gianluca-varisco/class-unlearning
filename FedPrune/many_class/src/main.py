#!/usr/bin/env python3

import time
import wandb
import torch
import numpy as np
import random

from config import (SEED, NUM_CLIENTS, NUM_ROUNDS, FL_LR, BATCH_SIZE,
                    LOCAL_EPOCHS, DEVICE)
from fl_training import run_fl_training
from class_unlearning import fuia_multiclass_unlearning_attack
from fed_prune_unlearning import cdp_unlearning


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_user_target_classes():
    print("=" * 60)
    print(" CONFIGURAZIONE FEDPRUNE MULTI-CLASS UNLEARNING")
    print("=" * 60)
    
    while True:
        try:
            k_input = input("Quante classi vuoi rimuovere con FedPrune? (2-4): ").strip()
            num_classes = int(k_input)
            if 2 <= num_classes <= 4:
                break
            print(">> Errore: inserisci un numero tra 2 e 4.")
        except ValueError:
            print(">> Errore: input non valido.")

    while True:
        try:
            raw_input = input(f"Inserisci le {num_classes} classi separate da spazio (es. 2 4 6): ").strip()
            classes = [int(c) for c in raw_input.replace(",", " ").split()]
            classes = list(set(classes))
            if len(classes) == num_classes and all(0 <= c <= 9 for c in classes):
                return sorted(classes)
            print(f">> Errore: Inserisci esattamente {num_classes} cifre distinte tra 0 e 9.")
        except ValueError:
            print(">> Errore: formato non valido.")


if __name__ == "__main__":
    target_classes = get_user_target_classes()
    k_targets = len(target_classes)
    prune_ratio = 0.3        # Quota canali convoluzionali da potare
    fine_tune_rounds = 3     # 3 round invece di 50-80
    beta_param = 0.5

    set_seed(SEED)

    wandb.init(project="FUIA_Class_Unlearning", config={
        "scenario": "multi_class_unlearning_fedprune",
        "dataset": "MNIST",
        "unlearning_method": "fedprune_cdp",
        "num_clients": NUM_CLIENTS,
        "seed": SEED,
        "target_classes": target_classes,
        "num_unlearned_classes": k_targets,
        "prune_ratio": prune_ratio,
        "fine_tune_rounds": fine_tune_rounds
    })

    total_t0 = time.time()

    # ------------------------------------------------------------
    # Fase 1: Federated Learning Completo (W^o)
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Fase 1: Federated Learning (Modello Originale W^o)")
    print("-" * 60)
    (original_model, stored_updates, client_data, private_data,
     pretrained_sd, round_selections, test_loader) = run_fl_training()

    # ------------------------------------------------------------
    # Fase 2: Multi-Class Unlearning con FedPrune (CDP + Fine-Tuning)
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"Fase 2: FedPrune Multi-Class Unlearning (Classi: {target_classes})")
    print("-" * 60)
    t_unlearn_0 = time.time()
    unlearned_model = cdp_unlearning(
        original_model=original_model,
        private_data=private_data,
        client_data=client_data,
        target_classes=target_classes,
        round_selections=round_selections,
        prune_ratio=prune_ratio,
        fine_tune_rounds=fine_tune_rounds
    )
    unlearn_time = time.time() - t_unlearn_0
    print(f"[FedPrune] Completato in {unlearn_time:.1f}s")

    # ------------------------------------------------------------
    # Fase 3: Attacco FUIA Multi-Class (Top-k Score S_d)
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"Fase 3: Attacco FUIA Multi-Class (Top-{k_targets} Score S_d)")
    print("-" * 60)
    predicted_classes, scores = fuia_multiclass_unlearning_attack(
        original_model, unlearned_model, top_k=k_targets, beta=beta_param
    )

    print("\n" + "=" * 60)
    print(f" RISULTATI ATTACCO FUIA SU FEDPRUNE (N_uc = {k_targets})")
    print("=" * 60)
    for c_id, score in enumerate(scores):
        mark = "<-- TOP CLASSE PREDETTA" if c_id in predicted_classes else ""
        print(f"  Classe {c_id}: Score S_d = {score:.4f} {mark}")

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
        "unlearn_time_s": unlearn_time,
        "total_time_s": total_time
    })
    wandb.finish()
