#!/usr/bin/env python3

import time
import wandb

from config import (SEED, NUM_CLIENTS, FRACTION, NUM_ROUNDS, LOCAL_EPOCHS,
                    BATCH_SIZE, FL_LR, PRETRAIN_EPOCHS, PRETRAIN_LR, NUM_CLASSES,
                    DATA_PER_CLIENT, INV_RESTARTS)
from fl_training import run_fl_training
from class_unlearning import retrain_without_class, fuia_class_unlearning_attack


if __name__ == "__main__":
    wandb.init(project="FUIA_Class_Unlearning", config={
        "scenario": "class_unlearning",
        "dataset": "MNIST",
        "unlearning_method": "retraining",
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "seed": SEED,
    })

    total_t0 = time.time()

    # Fase 1: Federated Learning Completo
    print("-" * 60)
    print("Fase 1: Federated Learning (Modello Originale W^o)")
    print("-" * 60)
    (original_model, stored_updates, client_data, private_data,
     pretrained_sd, round_selections, test_loader) = run_fl_training()

    # Classe da rimuovere per il test (es. la cifra '3')
    target_class_to_remove = 3
    beta_param = 0.5

    # Fase 2: Class Unlearning (Retraining senza la classe target)
    print("\n" + "-" * 60)
    print(f"Fase 2: Class Unlearning (Retraining senza classe {target_class_to_remove})")
    print("-" * 60)
    unlearned_model = retrain_without_class(
        pretrained_sd, private_data, client_data, target_class_to_remove, round_selections
    )

    # Fase 3: Attacco FUIA Class Unlearning
    print("\n" + "-" * 60)
    print("Fase 3: Attacco FUIA Class Unlearning (Calcolo Score S_d)")
    print("-" * 60)
    predicted_class, scores = fuia_class_unlearning_attack(
        original_model, unlearned_model, beta=beta_param
    )

    print("\n" + "=" * 60)
    print(" RISULTATI ATTACCO CLASS UNLEARNING")
    print("=" * 60)
    for c_id, score in enumerate(scores):
        mark = "<-- CLASSE PREDETTA DALL'ATTACCO" if c_id == predicted_class else ""
        print(f"  Classe {c_id}: Score S_d = {score:.4f} {mark}")

    is_success = (predicted_class == target_class_to_remove)
    print("-" * 60)
    print(f"  Classe Rimossa Reale:        {target_class_to_remove}")
    print(f"  Classe Predetta dall'Attacco: {predicted_class}")
    print(f"  Esito Attacco:                {'SUCCESS' if is_success else 'FAILED'}")
    print("=" * 60)

    total_time = time.time() - total_t0
    print(f"\nTempo totale: {total_time:.0f}s ({total_time / 60:.1f} min)")
    wandb.log({
        "attack/success": is_success,
        "attack/predicted_class": predicted_class,
        "attack/target_class": target_class_to_remove,
        "total_time_s": total_time
    })
    wandb.finish()