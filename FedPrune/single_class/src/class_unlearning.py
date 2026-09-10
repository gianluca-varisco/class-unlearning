import torch


def fuia_class_unlearning_attack(original_model, unlearned_model, beta=0.5):
    """
    Algoritmo 3 del paper FUIA (Zhou et al.): Class Unlearning Identification.
    Calcola lo Score di Discriminazione Sd[i] per ogni classe i per identificare
    quale classe è stata rimossa durante l'unlearning.
    """
    state_o = original_model.state_dict()
    state_u = unlearned_model.state_dict()

    # Estrazione dei pesi e dei bias dell'ultimo layer lineare (classifier.2)
    w_o = state_o['classifier.2.weight'].float()  # Shape: [10, 512]
    b_o = state_o['classifier.2.bias'].float()    # Shape: [10]

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

    # Identificazione della singola classe con lo score massimo
    predicted_class = torch.argmax(S_d).item()

    return predicted_class, S_d.numpy()
