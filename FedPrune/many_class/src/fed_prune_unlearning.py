import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from config import (
    BATCH_SIZE,
    DEVICE,
    FL_LR,
    LOCAL_EPOCHS,
    NUM_CLASSES,
    NUM_ROUNDS,
)
from federated import client_update, fedavg, lr_schedule
from model import CNN


def compute_multiclass_channel_activations(model, dataloader, target_classes, device):
    """
    Calcola l'attivazione media dei canali convoluzionali per i campioni
    appartenenti all'insieme target_classes (Wang et al., WWW 2022).
    """
    model.eval()
    conv1_acts = []
    conv2_acts = []

    def hook_conv1(module, input, output):
        conv1_acts.append(output.detach())

    def hook_conv2(module, input, output):
        conv2_acts.append(output.detach())

    # features[0] = Conv2d(1, 32), features[3] = Conv2d(32, 64) in model.py
    h1 = model.features[0].register_forward_hook(hook_conv1)
    h2 = model.features[3].register_forward_hook(hook_conv2)

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            # Maschera booleana per identificare i campioni di una qualsiasi classe target
            mask = torch.zeros_like(target, dtype=torch.bool)
            for c in target_classes:
                mask |= (target == c)

            if mask.sum() == 0:
                continue
            _ = model(data[mask])

    h1.remove()
    h2.remove()

    if not conv1_acts or not conv2_acts:
        return None, None

    all_c1 = torch.cat(conv1_acts, dim=0).mean(dim=[0, 2, 3])
    all_c2 = torch.cat(conv2_acts, dim=0).mean(dim=[0, 2, 3])

    return all_c1, all_c2


def apply_multiclass_pruning(model, target_classes, dataloader, prune_ratio=0.3, device=DEVICE):
    """
    Applica il pruning azzerando i filtri convoluzionali più attivi sulle classi rimosse
    e azzerando le righe corrispondenti nel classificatore lineare finale.
    """
    pruned_model = copy.deepcopy(model)
    pruned_model.to(device)

    c1_act, c2_act = compute_multiclass_channel_activations(
        pruned_model, dataloader, target_classes, device
    )

    with torch.no_grad():
        # 1. Pruning sui filtri di features[0] (Conv2d: 32 filtri)
        if c1_act is not None:
            k1 = max(1, int(len(c1_act) * prune_ratio))
            top_k_c1 = torch.topk(c1_act, k1).indices
            for idx in top_k_c1:
                pruned_model.features[0].weight[idx].zero_()
                if pruned_model.features[0].bias is not None:
                    pruned_model.features[0].bias[idx].zero_()

        # 2. Pruning sui filtri di features[3] (Conv2d: 64 filtri)
        if c2_act is not None:
            k2 = max(1, int(len(c2_act) * prune_ratio))
            top_k_c2 = torch.topk(c2_act, k2).indices
            for idx in top_k_c2:
                pruned_model.features[3].weight[idx].zero_()
                if pruned_model.features[3].bias is not None:
                    pruned_model.features[3].bias[idx].zero_()

        # 3. Azzeramento dei neuroni di classificazione per tutte le classi target
        for c in target_classes:
            pruned_model.classifier[2].weight[c].zero_()
            pruned_model.classifier[2].bias[c].zero_()

    return pruned_model


def cdp_unlearning(original_model, private_data, client_data, target_classes,
                   round_selections, prune_ratio=0.3, fine_tune_rounds=3):
    """
    Pipeline FedPrune / CDP (Wang et al., WWW 2022) per Single e Multi-Class:
    - Pruning selettivo basato sull'insieme target_classes
    - Fine-tuning federato rapido (pochi round) sui soli dati rimasti
    """
    if isinstance(target_classes, int):
        target_classes = [target_classes]

    # Campioni appartenenti a qualsiasi classe da rimuovere
    target_indices = [
        i for i, (_, label) in enumerate(private_data) if label in target_classes
    ]
    target_loader = DataLoader(
        Subset(private_data, target_indices),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"\n[FedPrune] 1. Pruning selettivo filtri per Classi {target_classes} (Ratio: {prune_ratio})...")
    unlearned_model = apply_multiclass_pruning(
        original_model, target_classes, target_loader, prune_ratio=prune_ratio, device=DEVICE
    )

    # Dataset trattenuto (senza le classi rimosse)
    retained_indices = [
        i for i, (_, label) in enumerate(private_data) if label not in target_classes
    ]
    retained_client_data = {
        c_id: [idx for idx in idx_list if idx in retained_indices]
        for c_id, idx_list in client_data.items()
    }

    print(f"[FedPrune] 2. Fine-tuning federato rapido ({fine_tune_rounds} round)...")
    unlearned_model.train()

    for rnd in range(1, fine_tune_rounds + 1):
        selected = round_selections[rnd]
        lr = FL_LR * 0.5
        results = []

        for k in selected:
            if not retained_client_data[k]:
                continue
            local = copy.deepcopy(unlearned_model)
            sd, n_k, loss = client_update(
                local, private_data, retained_client_data[k],
                LOCAL_EPOCHS, BATCH_SIZE, lr, DEVICE
            )
            results.append((sd, n_k, loss))

        if results:
            unlearned_model = fedavg(unlearned_model, results)

            # Mantiene a zero le righe del classificatore per tutte le classi rimosse
            with torch.no_grad():
                for c in target_classes:
                    unlearned_model.classifier[2].weight[c].zero_()
                    unlearned_model.classifier[2].bias[c].zero_()

    return unlearned_model