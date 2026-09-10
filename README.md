# Federated Unlearning & Model Inversion Attack (FUIA)

Questo repository contiene l'implementazione, l'estensione e la validazione sperimentale dell'attacco **FUIA (Federated Unlearning Inversion Attack)**, proposto nel paper:

> **"Model Inversion Attack Against Federated Unlearning"**  
> *L. Zhou, Y. Zhu, R. Liu* — IEEE Transactions on Information Forensics and Security, 2026.

Il progetto rappresenta la prosecuzione e l'evoluzione del lavoro precedentemente svolto da **Ottone Piazzi**, estendendo l'analisi empirica e confrontando metodologie sia di **Exact Unlearning** (Retraining da zero) sia di **Approximate Unlearning** basato su potatura dei canali (**FedPrune / Class-Discriminative Pruning - CDP**, Wang et al., *ACM WWW 2022*).

---

## 📌 Panoramica del Lavoro Svolto

Il focus principale è analizzare la vulnerabilità dei meccanismi di **Federated Unlearning** contro attacchi di *Model Inversion* e *Class Discrimination*, confrontando i modelli originale ($W^o$) e disimparato ($W^u$).

### 1. Prosecuzione ed Estensione del Lavoro di Piazzi
* Revisione della pipeline iniziale e correzione dell'interazione con il pre-training centralizzato (che fissava i pesi impedendo un corretto unlearning federato nel retraining).
* Transizione da un regime a volume minimo (`DATA_PER_CLIENT = 1`) a un setup scalabile e realistico (`DATA_PER_CLIENT = 100`), portando l'accuratezza del modello globale dal $56\%$ a oltre il **$96\%-99\%$** e stabilizzando i gradienti di divergenza.

### 2. Validazione Single-Class Unlearning (Retraining vs. FedPrune)
* **Retraining da Zero ($N_{\mathrm{uc}} = 1$):** Raggiunto il **100% di successo** nell'identificazione della singola classe rimossa tramite lo Score di Discriminazione $S_d[i]$ (Algoritmo 3 di FUIA).
* **FedPrune Single-Class:** Implementazione del pruning discriminativo sui canali convoluzionali (`features[0]` e `features[3]`) seguito da soli 3 round federati di fine-tuning. L'attacco ha ottenuto il **100% di successo** ($S_d = 0.4491$ senza pre-training e $S_d = 0.7344$ con pre-training) riducendo il tempo di unlearning da ~180s a soli **3.2–7.5s**.

### 3. Estensione Multi-Class Unlearning ($N_{\mathrm{uc}} \in \{2, 3, 4\}$)
* Generalizzazione dell'attacco FUIA mediante ranking **Top-k** su $S_d[i]$ per l'identificazione contemporanea di più classi eliminate.
* **Retraining Multi-Classe:** Validazione empirica della Table I del paper di Zhou et al.:
  * $N_{\mathrm{uc}} = 2$: $100\%$ Full Success ($N_{\mathrm{ic}} = 2/2$).
  * $N_{\mathrm{uc}} = 3$: $100\%$ Full Success ($N_{\mathrm{ic}} = 3/3$).
  * $N_{\mathrm{uc}} = 4$: Verifica della degradazione teorica causata dal *backbone drift* ($N_{\mathrm{ic}} = 3/4$, esito parziale documentato dagli autori).
* **FedPrune Multi-Classe:** Implementazione del pruning cumulativo congiunto e azzeramento deterministico dei rispettivi neuroni di classificazione:
  * $N_{\mathrm{uc}} = 2$ (Classi `[3, 5]`): **100% Full Success** ($N_{\mathrm{ic}} = 2/2$) in **3.4s**.
  * $N_{\mathrm{uc}} = 3$ (Classi `[1, 4, 5]`): **100% Full Success** ($N_{\mathrm{ic}} = 3/3$) in **3.6s**.
  * $N_{\mathrm{uc}} = 4$ (Classi `[2, 5, 6, 7]`): **100% Full Success** ($N_{\mathrm{ic}} = 4/4$) in **4.0s**. A differenza del retraining, l'assenza di drift sul backbone permette a FUIA di raggiungere il pieno successo anche con 4 classi rimosse.

---

## 📊 Sintesi dei Risultati Sperimentali

| Metodologia | Scenario ($N_{\mathrm{uc}}$) | Classi Rimosse | Classi Identificate ($N_{\mathrm{ic}}$) | Esito Attacco FUIA | Tempo di Unlearning |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Retraining** | Single-Class (1) | `[3]` | 1 su 1 | **FULL SUCCESS (100%)** | ~180s |
| **Retraining** | Multi-Class (2) | `[3, 7]` | 2 su 2 | **FULL SUCCESS (100%)** | ~180s |
| **Retraining** | Multi-Class (3) | `[1, 4, 8]` | 3 su 3 | **FULL SUCCESS (100%)** | ~180s |
| **Retraining** | Multi-Class (4) | `[0, 2, 6, 9]` | 3 su 4 | **PARTIAL SUCCESS (75%)** | ~180s |
| **FedPrune (CDP)** | Single-Class (1) | `[3]` | 1 su 1 | **FULL SUCCESS (100%)** | **3.2s – 7.5s** |
| **FedPrune (CDP)** | Multi-Class (2) | `[3, 5]` | 2 su 2 | **FULL SUCCESS (100%)** | **3.4s** |
| **FedPrune (CDP)** | Multi-Class (3) | `[1, 4, 5]` | 3 su 3 | **FULL SUCCESS (100%)** | **3.6s** |
| **FedPrune (CDP)** | Multi-Class (4) | `[2, 5, 6, 7]` | 4 su 4 | **FULL SUCCESS (100%)** | **4.0s** |

---

## 💡 Riscontri Teorici e Metodologici

1. **Trade-off Efficienza Computazionale:**  
   FedPrune riduce il tempo di unlearning di oltre il **$96\%$** rispetto al retraining da zero (da svariati minuti a circa 3–4 secondi) intervenendo direttamente a posteriori su $W^o$ tramite potatura e soli 3 round di fine-tuning.
2. **Esposizione di Privacy Intrinseca:**  
   L'azzeramento forzato dei parametri del classificatore ($\Vert w_o[c] - 0 \Vert_1 = \Vert w_o[c] \Vert_1$) massimizza la divergenza differenziale, rendendo le classi dimenticate ancora più visibili e isolate rispetto al rumore di fondo delle classi trattenute.
3. **Resilienza dell'Attacco a $N_{\mathrm{uc}} = 4$:**  
   Mentre nel Retraining la riorganizzazione geometrica dei filtri condivisi (*backbone drift*) degrada l'identificazione al 75%, in FedPrune la struttura del modello rimane stabile, consentendo a FUIA di raggiungere il **100% di accuratezza anche su 4 classi eliminate congiuntamente**.

---

##  Tecnologie Utilizzate

* **PyTorch** & **Torchvision** (Deep Learning & CNN)
* **Federated Averaging (FedAvg)** (Simulazione FL multi-client)
* **Weights & Biases (wandb)** (Tracking sperimentale)

##  Riferimenti Bibliografici e Lavori Correlati

- L. Zhou, Y. Zhu, and R. Liu, *"Model Inversion Attack Against Federated Unlearning"*, IEEE Transactions on Information Forensics and Security, vol. 21, pp. 2342–2357, 2026. https://ieeexplore.ieee.org/document/11400570
- J. Wang, S. Guo, X. Xie, and H. Qi, *"Federated Unlearning via Class-Discriminative Pruning"*, in Proceedings of the ACM Web Conference 2022 (WWW '22), pp. 622–632, 2022. https://arxiv.org/pdf/2110.11794
- N. Romandini, A. Mora, C. Mazzocca, R. Montanari, and P. Bellavista, *"Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics"*, IEEE Transactions on Neural Networks and Learning Systems, 2024. https://doi.org/10.1109/TNNLS.2024.3478334
- O. Piazzi, *"Analisi e implementazione di attacchi per ricostruzione dati nel Federated Unlearning"*, Relazione di Tirocinio. https://github.com/ottonepiazzi/federated-learning


## Autore

**Gianluca Varisco** — [github.com/gianluca-varisco](https://github.com/gianluca-varisco)
