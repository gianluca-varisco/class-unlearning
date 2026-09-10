import torch


def fuia_multiclass_unlearning_attack(original_model, unlearned_model, top_k=1, beta=0.5):
    """
    Algoritmo 3 del paper FUIA (Zhou et al., 2026) esteso a Multi-Class Unlearning (Top-k):
    Calcola lo Score di Discriminazione S_d[i] per ogni classe i (Eqg. 20-21)
    ed estrae i primi top_k indici con il punteggio più alto.
    """
    state_o = original_model.state_dict()
    state_u = unlearned_model.state_dict()

    # Estrazione parametri dell'ultimo layer lineare (classifier.2)
    w_o = state_o['classifier.2.weight'].float()  # Shape: [10, 512]
    b_o = state_o['classifier.2.bias'].float()    # Shape: [10]

    w_u = state_u['classifier.2.weight'].float()
    b_u = state_u['classifier.2.bias'].float()

    num_classes = w_o.shape[0]

    # Calcolo differenze norma-L1 dei pesi e valore assoluto dei bias per classe (Eq. 20)
    v_diff = torch.zeros(num_classes)
    b_diff = torch.zeros(num_classes)

    for i in range(num_classes):
        v_diff[i] = torch.norm(w_o[i] - w_u[i], p=1)
        b_diff[i] = torch.abs(b_o[i] - b_u[i])

    # Normalizzazione con epsilon per stabilità numerica
    v_sum = torch.sum(v_diff) + 1e-12
    b_sum = torch.sum(b_diff) + 1e-12

    # Score di discriminazione S_d[i] per ciascuna classe (Eq. 21)
    S_d = beta * (v_diff / v_sum) + (1.0 - beta) * (b_diff / b_sum)

    # Selezione dei primi top_k punteggi massimi
    _, topk_indices = torch.topk(S_d, k=top_k)
    predicted_classes = sorted(topk_indices.tolist())

    return predicted_classes, S_d.cpu().numpy()


def fuia_class_unlearning_attack(original_model, unlearned_model, beta=0.5):
    """
    Compatibilità per scenari a classe singola (top_k=1).
    """
    predicted_classes, scores = fuia_multiclass_unlearning_attack(
        original_model, unlearned_model, top_k=1, beta=beta
    )
    return predicted_classes[0], scores
