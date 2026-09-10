# Analisi Comparativa: Multi-Class Federated Unlearning via FedPrune e Vulnerabilità all'Attacco FUIA

Questo documento raccoglie, analizza e documenta i riscontri empirici derivanti dalla validazione dell'algoritmo **FedPrune / Class-Discriminative Pruning (CDP)** (Wang et al., *ACM WWW 2022*) esteso allo scenario **Multi-Class Unlearning** ($N_{\mathrm{uc}} \in \{2, 3, 4\}$) su dataset MNIST, confrontando la sua resilienza all'attacco **FUIA (Top-$k$ Ranking)** (Zhou et al., *IEEE TIFS 2026*) rispetto al **Retraining esatto da zero**[cite: 1, 3, 6].

---

## 1. Dati Sperimentali Raccolti

Tutti i test condividono la medesima architettura convoluzionale (CNN a 2 stadi convoluzionali e testa fully-connected), $K=50$ client con regime ad alto volume di campioni e un modello originale $W^o$ convergente con accuratezza finale del **$99.2\%$**[cite: 3, 4, 6].

### Test 1: $N_{\mathrm{uc}} = 2$ (Classi Target: `[3, 5]`)
* **Tempo di calcolo FedPrune:** **3.4 secondi** (Pruning + 3 round di fine-tuning)
* **Score di Discriminazione $S_d$:**
  * Classe 5: $\mathbf{0.6544}$ (Top-1)
  * Classe 3: $\mathbf{0.3163}$ (Top-2)
  * Restanti classi trattenute: comprese tra $0.0006$ e $0.0143$
* **Massa di divergenza assorbita dalle classi target:** **$97.07\%$**
* **Esito Attacco FUIA:** **2 su 2 identificate (FULL SUCCESS - 100%)**

---

### Test 2: $N_{\mathrm{uc}} = 3$ (Classi Target: `[1, 4, 5]`)
* **Tempo di calcolo FedPrune:** **3.6 secondi**
* **Score di Discriminazione $S_d$:**
  * Classe 4: $\mathbf{0.4835}$ (Top-1)
  * Classe 1: $\mathbf{0.2589}$ (Top-2)
  * Classe 5: $\mathbf{0.2448}$ (Top-3)
  * Restanti classi trattenute: comprese tra $0.0001$ e $0.0064$
* **Massa di divergenza assorbita dalle classi target:** **$98.72\%$**
* **Esito Attacco FUIA:** **3 su 3 identificate (FULL SUCCESS - 100%)**

---

### Test 3: $N_{\mathrm{uc}} = 4$ (Classi Target: `[2, 5, 6, 7]`)
* **Tempo di calcolo FedPrune:** **4.0 secondi**
* **Score di Discriminazione $S_d$:**
  * Classe 6: $\mathbf{0.3661}$ (Top-1)
  * Classe 7: $\mathbf{0.2962}$ (Top-2)
  * Classe 2: $\mathbf{0.1950}$ (Top-3)
  * Classe 5: $\mathbf{0.1425}$ (Top-4)
  * Restanti classi trattenute: comprese tra $0.0000$ e $0.0001$
* **Massa di divergenza assorbita dalle classi target:** **$99.98\%$**
* **Esito Attacco FUIA:** **4 su 4 identificate (FULL SUCCESS - 100%)**

---

## 2. Sintesi Comparativa dei Risultati

| Scenario ($N_{\mathrm{uc}}$) | Metodo di Unlearning | Classi Target ($C_{\mathrm{target}}$) | Classi Identificate ($N_{\mathrm{ic}}$) | Esito Attacco FUIA | Tempo Unlearning |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2 Classi** | Retraining da zero[cite: 3] | `[3, 7]`[cite: 2] | 2 su 2[cite: 3] | **FULL SUCCESS (100%)**[cite: 3] | ~180s (minuti)[cite: 1] |
| **2 Classi** | **FedPrune (CDP)** | `[3, 5]` | **2 su 2** | **FULL SUCCESS (100%)** | **3.4s** |
| **3 Classi** | Retraining da zero[cite: 3] | `[1, 4, 8]` | 3 su 3[cite: 3] | **FULL SUCCESS (100%)**[cite: 3] | ~180s (minuti)[cite: 1] |
| **3 Classi** | **FedPrune (CDP)** | `[1, 4, 5]` | **3 su 3** | **FULL SUCCESS (100%)** | **3.6s** |
| **4 Classi** | Retraining da zero[cite: 3] | `[0, 2, 6, 9]` | 3 su 4[cite: 3] | *PARTIAL SUCCESS (75%)*[cite: 3] | ~180s (minuti)[cite: 1] |
| **4 Classi** | **FedPrune (CDP)** | `[2, 5, 6, 7]` | **4 su 4** | **FULL SUCCESS (100%)** | **4.0s** |

---

## 3. Analisi Teorica delle Differenze: Perché FedPrune Raggiunge il 100% a 4 Classi

Uno dei riscontri scientifici di maggior rilievo è la discrepanza osservata nello scenario a **4 classi rimosse ($N_{\mathrm{uc}} = 4$)**:

### A. La Degradazione Teorica nel Retraining (*Backbone Drift*)
Nel paper di Zhou et al. (*Table I*), gli autori documentano una degradazione delle prestazioni di identificazione quando $N_{\mathrm{uc}} = 4$, replicata nei nostri test di retraining (3 su 4 corrette)[cite: 3].
* **Causa:** Rimuovere il $40\%$ delle classi costringe l'addestramento da zero a convergere su un'ipersfera di parametri radicalmente diversa[cite: 1]. I filtri convoluzionali del backbone si riorganizzano per ottimizzare solo le 6 classi residue, provocando un forte scostamento casuale anche sui pesi delle classi trattenute[cite: 1, 4].
* Questo *backbone drift* alza il rumore di fondo dei canali non target, portando uno dei pesi rimasti a superare accidentalmente una delle classi eliminate nel ranking normalizzato $S_d$[cite: 1].

### B. La Robustezza Deterministica di FedPrune
In FedPrune, il fenomeno del *backbone drift* non si manifesta, consentendo a FUIA di raggiungere il **$100\%$ di successo anche con 4 classi rimosse** ($S_d$ congiunto $= 0.9998$):
1. **Preservazione delle Feature Comuni:** Il modello non viene riaddestrato da zero; la maggior parte dei pesi convoluzionali e i neuroni delle classi mantenute rimangono ancorati alla soluzione ottima già raggiunta in $W^o$[cite: 6].
2. **Hard Zero-Out del Classificatore:** Per tutte le classi $c \in C_{\mathrm{target}}$, i parametri dell'ultimo strato vengono azzerati esplicitamente:
   $$W_{\text{classifier}}[c] \leftarrow 0, \quad b_{\text{classifier}}[c] \leftarrow 0 \quad \forall c \in C_{\mathrm{target}}$$[cite: 4, 7]
3. **Stabilità nel Fine-Tuning:** Durante i soli 3 round di federazione, le 6 classi conservate non subiscono deriva geometrica ma solo minimi aggiustamenti[cite: 6, 7].
4. Di conseguenza, le differenze norma-$L_1$ per le classi eliminate corrispondono esattamente all'intera magnitudo dei pesi originali ($\Delta w_c = \|w_o[c]\|_1$), mentre per le classi mantenute rimangono prossime a zero ($S_d \le 0.0001$)[cite: 1].

---

## 4. Conclusioni Metodologiche

1. **Trade-off Efficienza/Privacy:** FedPrune risolve il collo di bottiglia temporale del federated unlearning multi-classe, riducendo la latenza da svariati minuti a soli **3-4 secondi** (un'accelerazione di circa $45\times$)[cite: 1, 6].
2. **Esposizione di Sicurezza Intrinseca:** L'intervento chirurgico post-training basato su azzeramento dei pesi non solo non scherma il modello da attacchi di inferenza, ma rende il confine differenziale tra classi dimenticate e trattenute ancora più marcato rispetto al retraining stocastico, rendendo **FedPrune estremamente vulnerabile a FUIA su qualsiasi cardinalità di classi eliminate**[cite: 1, 6].

---

## 5. Riferimenti Bibliografici

- J. Wang, S. Guo, X. Xie, and H. Qi, *"Federated Unlearning via Class-Discriminative Pruning"*, in Proceedings of the ACM Web Conference 2022 (WWW '22), pp. 622–632, 2022. https://arxiv.org/abs/2110.11794 (PDF: https://arxiv.org/pdf/2110.11794)
- L. Zhou, Y. Zhu, and R. Liu, *"Model Inversion Attack Against Federated Unlearning"*, IEEE Transactions on Information Forensics and Security, vol. 21, pp. 2342–2357, 2026. https://ieeexplore.ieee.org/document/11400570
- N. Romandini, A. Mora, C. Mazzocca, R. Montanari, and P. Bellavista, *"Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics"*, IEEE Transactions on Neural Networks and Learning Systems, 2024. https://doi.org/10.1109/TNNLS.2024.3478334
- O. Piazzi, *"Analisi e implementazione di attacchi per ricostruzione dati nel Federated Unlearning"*, Relazione di Tirocinio. https://github.com/ottonepiazzi/federated-learning
