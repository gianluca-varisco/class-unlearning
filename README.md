# Federated Unlearning & Model Inversion Attack (FUIA)

Questo repository contiene l'implementazione, l'estensione e la validazione sperimentale dell'attacco **FUIA (Federated Unlearning Inversion Attack)**, proposto nel paper:

> **"Model Inversion Attack Against Federated Unlearning"**  
> *L. Zhou, Y. Zhu, R. Liu* — IEEE Transactions on Information Forensics and Security, 2026.

Il progetto rappresenta la prosecuzione e l'evoluzione del lavoro precedentemente svolto da **Ottone Piazzi**, estendendo l'analisi empirica e risolvendo i vincoli metodologici legati agli scenari di unlearning in Federated Learning (FL).

---

##  Panoramica del Lavoro Svolto

Il focus principale è stato analizzare la vulnerabilità dei meccanismi di **Federated Unlearning via Retraining** contro attacchi di *Model Inversion* e *Class Discrimination*, confrontando i modelli originale ($W^o$) e disimparato ($W^u$).

### 1. Prosecuzione ed Estensione del Lavoro di Piazzi
* Revisione della pipeline iniziale e correzione dell'interazione con il pre-training centralizzato (che fissava i pesi impedendo un corretto unlearning federato).
* Transizione da un regime a volume minimo (`DATA_PER_CLIENT = 1`) a un setup scalabile e realistico (`DATA_PER_CLIENT = 100`), portando l'accuratezza del modello dal $56\%$ al **$96.2\%$** e stabilizzando i gradienti di divergenza.

### 2. Validazione Single-Class Unlearning
* Implementazione e verifica dell'Algoritmo 3 per l'isolamento di una singola classe target rimossa ($N_{\mathrm{uc}} = 1$).
* Raggiunto il **100% di successo** nell'identificazione della classe dimenticata tramite lo Score di Discriminazione $S_d[i]$.

### 3. Estensione Multi-Class Unlearning ($N_{\mathrm{uc}} \in \{2, 3, 4\}$)
* Generalizzazione dell'attacco FUIA mediante ranking **Top-$k$** su $S_d[i]$ per l'identificazione contemporanea di più classi eliminate.
* Validazione empirica della **Table I** del paper:
  * **2 classi rimosse ($N_{\mathrm{uc}} = 2$):** $100\%$ Full Success ($N_{\mathrm{ic}} = 2/2$).
  * **3 classi rimosse ($N_{\mathrm{uc}} = 3$):** $100\%$ Full Success ($N_{\mathrm{ic}} = 3/3$).
  * **4 classi rimosse ($N_{\mathrm{uc}} = 4$):** Verifica della degradazione teorica causata dal *backbone drift* ($N_{\mathrm{ic}} = 3/4$, esito parziale documentato dagli autori).

---

##  Sintesi dei Risultati Sperimentali

| Scenario | Classi Rimosse ($N_{\mathrm{uc}}$) | Classi Identificate ($N_{\mathrm{ic}}$) | Esito Attacco | Accuracy $W^o \rightarrow W^u$ |
| :--- | :---: | :---: | :---: | :---: |
| **Single-Class** | 1 | 1 | **FULL SUCCESS (100%)** | $96.2\% \rightarrow 86.5\%$ |
| **Multi-Class (2)** | 2 | 2 | **FULL SUCCESS (100%)** | $96.2\% \rightarrow 76.7\%$ |
| **Multi-Class (3)** | 3 | 3 | **FULL SUCCESS (100%)** | $96.2\% \rightarrow 67.5\%$ |
| **Multi-Class (4)** | 4 | 3 | **PARTIAL SUCCESS (75%)** | $96.2\% \rightarrow 57.5\%$ |

---

##  Tecnologie Utilizzate

* **PyTorch** & **Torchvision** (Deep Learning & CNN)
* **Federated Averaging (FedAvg)** (Simulazione FL multi-client)
* **Weights & Biases (wandb)** (Tracking sperimentale)

##  Riferimenti Bibliografici e Lavori Correlati

1. **L. Zhou, Y. Zhu, and R. Liu**, *"Model Inversion Attack Against Federated Unlearning"*, IEEE Transactions on Information Forensics and Security, vol. 21, pp. 2342–2357, 2026.  
   [DOI: 10.1109/TIFS.2026.11400570](https://ieeexplore.ieee.org/document/11400570)

2. **N. Romandini, A. Mora, C. Mazzocca, R. Montanari, and P. Bellavista**, *"Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics"*, IEEE Transactions on Neural Networks and Learning Systems, 2024.  
   [DOI: 10.1109/TNNLS.2024.3478334](https://doi.org/10.1109/TNNLS.2024.3478334) — [arXiv:2401.05146](https://arxiv.org/abs/2401.05146)

3. **O. Piazzi**, *"Analisi e implementazione di attacchi per ricostruzione dati nel Federated Unlearning"*, Relazione di Tirocinio.  
   Repository GitHub: [ottonepiazzi/federated-learning](https://github.com/ottonepiazzi/federated-learning)
