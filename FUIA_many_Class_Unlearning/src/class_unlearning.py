import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import copy
import numpy as np

from config import NUM_ROUNDS, FL_LR, LOCAL_EPOCHS, BATCH_SIZE, DEVICE, NUM_CLASSES
from model import CNN
from federated import client_update, fedavg, lr_schedule, evaluate


def retrain_without_classes(pretrained_sd, private_data, client_data,
                            target_classes, round_selections, test_loader=None):
    """
    Esegue il Multi-Class Unlearning tramite Retraining:
    Addestra nuovamente il modello escludendo tutti i campioni appartenenti alle classi in 'target_classes'.
    """
    if isinstance(target_classes, int):
        target_classes = [target_classes]

    # 1. Filtriamo il dataset privato rimuovendo tutte le classi target
    retained_indices = [i for i, (_, label) in enumerate(private_data) if label not in target_classes]
    retained_private_data = Subset(private_data, retained_indices)

    # Mappiamo i dati dei client per considerare solo gli indici rimasti
    retained_client_data = {}
    for c_id, idx_list in client_data.items():
        retained_client_data[c_id] = [idx for idx in idx_list if idx in retained_indices]

    model = CNN()
    model.load_state_dict(pretrained_sd)

    for rnd in range(1, NUM_ROUNDS + 1):
        selected = round_selections[rnd]
        lr = lr_schedule(FL_LR, rnd, NUM_ROUNDS)
        results = []
        for k in selected:
            if not retained_client_data[k]:
                continue
            local = copy.deepcopy(model)
            sd, n_k, loss = client_update(local, private_data, retained_client_data[k],
                                          LOCAL_EPOCHS, BATCH_SIZE, lr, DEVICE)
            results.append((sd, n_k, loss))
        if results:
            model = fedavg(model, results)

        # Calcolo Loss ponderata
        total_samples = sum(n_k for _, n_k, _ in results)
        round_loss = sum(l * n_k for _, n_k, l in results) / total_samples if total_samples > 0 else 0.0

        # Calcolo Accuracy sul Test Set
        acc_str = ""
        if test_loader is not None:
            acc = evaluate(model, test_loader, DEVICE)
            acc_percent = acc * 100 if acc <= 1.0 else acc
            acc_str = f" | Acc: {acc_percent:.1f}%"

        if rnd == 1 or rnd % 10 == 0 or rnd == NUM_ROUNDS:
            print(f"  Round {rnd:2d}/{NUM_ROUNDS} | LR: {lr:.5f} | Loss: {round_loss:.4f}{acc_str}")

    return model


def fuia_multiclass_unlearning_attack(original_model, unlearned_model, top_k=1, beta=0.5):
    """
    Algoritmo 3 del paper FUIA esteso a Multi-Class Unlearning (Top-k):
    Calcola lo Score di Discriminazione Sd[i] per ogni classe i ed estrae
    i primi top_k indici con il punteggio più alto.
    """
    state_o = original_model.state_dict()
    state_u = unlearned_model.state_dict()

    w_o = state_o['classifier.2.weight'].float()  # Shape: [10, 512]
    b_o = state_o['classifier.2.bias'].float()    # Shape: [10]

    w_u = state_u['classifier.2.weight'].float()
    b_u = state_u['classifier.2.bias'].float()

    num_classes = w_o.shape[0]

    # Differenze L1 dei pesi e valore assoluto dei bias per classe (Eq. 20)
    v_diff = torch.zeros(num_classes)
    b_diff = torch.zeros(num_classes)

    for i in range(num_classes):
        v_diff[i] = torch.norm(w_o[i] - w_u[i], p=1)
        b_diff[i] = torch.abs(b_o[i] - b_u[i])

    v_sum = torch.sum(v_diff) + 1e-12
    b_sum = torch.sum(b_diff) + 1e-12

    # Score di discriminazione Sd[i] per ciascuna classe (Eq. 21)
    S_d = beta * (v_diff / v_sum) + (1 - beta) * (b_diff / b_sum)

    # Estrazione dei Top-k punteggi più alti
    topk_scores, topk_indices = torch.topk(S_d, k=top_k)
    predicted_classes = sorted(topk_indices.tolist())

    return predicted_classes, S_d.numpy()