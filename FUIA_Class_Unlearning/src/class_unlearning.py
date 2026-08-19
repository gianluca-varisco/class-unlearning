import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import copy
import numpy as np

from config import NUM_ROUNDS, FL_LR, LOCAL_EPOCHS, BATCH_SIZE, DEVICE, NUM_CLASSES
from model import CNN
from federated import client_update, fedavg, lr_schedule


def retrain_without_class(pretrained_sd, private_data, client_data,
                          target_class, round_selections):
    """
    Esegue il Class Unlearning tramite Retraining:
    Addestra nuovamente il modello escludendo tutti i campioni appartenenti a 'target_class'.
    """
    # 1. Filtriamo il dataset privato rimuovendo la classe target
    retained_indices = [i for i, (_, label) in enumerate(private_data) if label != target_class]
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
            # Se il client non ha più dati dopo la rimozione della classe, lo saltiamo
            if not retained_client_data[k]:
                continue
            local = copy.deepcopy(model)
            sd, n_k, loss = client_update(local, private_data, retained_client_data[k],
                                          LOCAL_EPOCHS, BATCH_SIZE, lr, DEVICE)
            results.append((sd, n_k, loss))
        if results:
            model = fedavg(model, results)

    return model


def fuia_class_unlearning_attack(original_model, unlearned_model, beta=0.5):
    """
    Algoritmo 3 del paper FUIA (Zhou et al.): Class Unlearning Identification.
    Calcola lo Score di Discriminazione Sd[i] per ogni classe i per identificare
    quale classe è stata rimossa durante l'unlearning.
    """
    state_o = original_model.state_dict()
    state_u = unlearned_model.state_dict()

    # Estrazione dei pesi e dei bias dell'ultimo layer (classifier.2)
    w_o = state_o['classifier.2.weight'].float() # Shape: [10, 512]
    b_o = state_o['classifier.2.bias'].float()   # Shape: [10]

    w_u = state_u['classifier.2.weight'].float()
    b_u = state_u['classifier.2.bias'].float()

    num_classes = w_o.shape[0]

    # Differenze L1 dei pesi e valore assoluto dei bias per classe (Eq. 20 del paper)
    v_diff = torch.zeros(num_classes)
    b_diff = torch.zeros(num_classes)

    for i in range(num_classes):
        v_diff[i] = torch.norm(w_o[i] - w_u[i], p=1)
        b_diff[i] = torch.abs(b_o[i] - b_u[i])

    # Somma totale per normalizzare
    v_sum = torch.sum(v_diff) + 1e-12
    b_sum = torch.sum(b_diff) + 1e-12

    # Score di discriminazione Sd[i] per ciascuna classe (Eq. 21 del paper)
    S_d = beta * (v_diff / v_sum) + (1 - beta) * (b_diff / b_sum)

    # Identificazione della classe con lo score massimo
    predicted_class = torch.argmax(S_d).item()

    return predicted_class, S_d.numpy()