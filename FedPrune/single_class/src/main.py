#!/usr/bin/env python3

import time
import wandb

from config import SEED, NUM_CLIENTS
from fl_training import run_fl_training
from class_unlearning import fuia_class_unlearning_attack
from fed_prune_unlearning import cdp_unlearning


if __name__ == "__main__":
    target_class_to_remove = 3   # Classe da dimenticare
    prune_ratio = 0.3            # Quota di filtri da potare
    fine_tune_rounds = 3         # Solo 3 round di recupero invece di 50-80
    beta_param = 0.5

    wandb.init(project="FUIA_Class_Unlearning", config={
        "scenario": "class_unlearning_fedprune",
        "dataset": "MNIST",
        "unlearning_method": "fedprune_cdp",
        "target_class": target_class_to_remove,
        "prune_ratio": prune_ratio,
        "fine_tune_rounds": fine_tune_rounds,
        "num_clients": NUM_CLIENTS,
        "seed": SEED,
    })

    total_t0 = time.time()

    # ------------------------------------------------------------
    # Fase 1: Federated Learning Completo (W^o)
    # ------------------------------------------------------------
    print("-" * 60)
    print("Fase 1: Federated Learning (Modello Originale W^o)")
    print("-" * 60)
    (original_model, stored_updates, client_data, private_data,
     pretrained_sd, round_selections, test_loader) = run_fl_training()

    # ------------------------------------------------------------
    # Fase 2: Unlearning con FedPrune (CDP + Fine-Tuning)
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"Fase 2: Class Unlearning via FedPrune (Classe {target_class_to_remove})")
    print("-" * 60)
    t_unlearn_0 = time.time()
    unlearned_model = cdp_unlearning(
        original_model=original_model,
        private_data=private_data,
        client_data=client_data,
        target_class=target_class_to_remove,
        round_selections=round_selections,
        prune_ratio=prune_ratio,
        fine_tune_rounds=fine_tune_rounds
    )
    unlearn_time = time.time() - t_unlearn_0
    print(f"[FedPrune] Terminato in {unlearn_time:.1f}s")

    # ------------------------------------------------------------
    # Fase 3: Attacco FUIA Class Unlearning
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Fase 3: Attacco FUIA Class Unlearning (Calcolo Score S_d)")
    print("-" * 60)
    predicted_class, scores = fuia_class_unlearning_attack(
        original_model, unlearned_model, beta=beta_param
    )

    print("\n" + "=" * 60)
    print(" RISULTATI ATTACCO FUIA SU FEDPRUNE (CDP)")
    print("=" * 60)
    for c_id, score in enumerate(scores):
        mark = "<-- TOP CLASSE PREDETTA" if c_id == predicted_class else ""
        print(f"  Classe {c_id}: Score S_d = {score:.4f} {mark}")

    is_success = (predicted_class == target_class_to_remove)
    print("-" * 60)
    print(f"  Classe Rimossa Reale:         {target_class_to_remove}")
    print(f"  Classe Predetta dall'Attacco: {predicted_class}")
    print(f"  Esito Attacco:                 {'SUCCESS' if is_success else 'FAILED'}")
    print("=" * 60)

    total_time = time.time() - total_t0
    print(f"\nTempo totale: {total_time:.0f}s ({total_time / 60:.1f} min)")

    wandb.log({
        "attack/success": is_success,
        "attack/predicted_class": predicted_class,
        "attack/target_class": target_class_to_remove,
        "unlearning_time_s": unlearn_time,
        "total_time_s": total_time
    })
    wandb.finish()