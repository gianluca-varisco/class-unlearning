# Analisi Comparativa dei Risultati: Federated Unlearning via FedPrune (CDP) e Attacco FUIA

Questo documento analizza i risultati empirici ottenuti dall'esecuzione di **FedPrune (Class-Discriminative Pruning - CDP)**, proposto da Wang et al. (*ACM WWW 2022*), nello scenario di **Single-Class Unlearning** su dataset MNIST, confrontando l'efficacia e le dinamiche rispetto all'approccio di riferimento basato su **Retraining da zero** (Zhou et al., *IEEE TIFS 2026*).

---

## 1. Dati Sperimentali Raccolti

I log di esecuzione registrano il comportamento del sistema in tre fasi sequenziali:

### Fase 1: Federated Learning (Modello Originale $W^o$)
* **Pre-training centralizzato:** Disattivato (`PRETRAIN_EPOCHS = 0`)[cite: 1]. Il tempo di setup è stato di soli **3 secondi**, con un'accuratezza iniziale pari all'**$8.9\%$** (distribuzione uniforme e non addestrata sulle 10 classi).
* **Addestramento Federato:** 80 round con 50 client (`FL_LR = 0.01`)[cite: 1].
* **Convergenza:** L'accuratezza è passata dal $37.0\%$ (Round 1) al $92.0\%$ (Round 10), fino a stabilizzarsi a un'accuratezza finale del **$97.0\%$** ($96.98\%$) con funzione di perdita pari a $0.0448$.

### Fase 2: Class Unlearning via FedPrune (Classe 3)
* **Metodologia:** Calcolo della sensibilità di canale sui blocchi convoluzionali (`features[0]` e `features[3]`), azzeramento selettivo dei filtri con risposta maggiore (`prune_ratio = 0.3`) e azzeramento deterministico della riga corrispondente nel classificatore (`classifier[2]`)[cite: 4, 8].
* **Recovery federato:** 3 round rapidi di fine-tuning sulle sole classi trattenute[cite: 8].
* **Tempo di esecuzione dell'Unlearning:** **7.5 secondi**.

### Fase 3: Attacco FUIA (Algoritmo 3)
Distribuzione normalizzata dello Score di Discriminazione $S_d[i]$ ($\beta = 0.5$)[cite: 8, 9]:

| Classe $i$ | Score di Discriminazione $S_d[i]$ | Esito / Note |
| :---: | :---: | :--- |
| Classe 0 | 0.0590 | Rumore residuo di convergenza |
| Classe 1 | 0.0725 | Rumore residuo di convergenza |
| Classe 2 | 0.0688 | Rumore residuo di convergenza |
| **Classe 3** | **0.4491** | **TOP CLASSE PREDETTA (Classe Rimossa Reale)** |
| Classe 4 | 0.0677 | Rumore residuo di convergenza |
| Classe 5 | 0.0823 | Rumore residuo di convergenza |
| Classe 6 | 0.0218 | Rumore residuo di convergenza |
| Classe 7 | 0.0247 | Rumore residuo di convergenza |
| Classe 8 | 0.0915 | Rumore residuo di convergenza |
| Classe 9 | 0.0626 | Rumore residuo di convergenza |

* **Classe Rimossa Reale:** 3[cite: 9]
* **Classe Predetta dall'Attacco:** 3[cite: 9]
* **Esito Attacco:** **`SUCCESS` (100% Top-1 Accuracy)**[cite: 9]

---

## 2. Il Ruolo del Pre-training: Perché Falliva nel Retraining e Funziona in FedPrune

Uno degli aspetti teorici e sperimentali più rilevanti emersi dal confronto riguarda l'impatto del **pre-training centralizzato** (`PRETRAIN_EPOCHS > 0`):

### A. Il Meccanismo del Fallimento nel Retraining
Nel Retraining da zero, sia il modello originale $W^o$ che il modello da disimparare $W^u$ partivano dallo stesso checkpoint pre-addestrato centralmente su 48.000 campioni (tutte e 10 le classi comprese)[cite: 1, 2].
* Durante il pre-training, l'ultimo layer lineare imparava in modo approfondito i pesi della classe target $c$[cite: 1, 2].
* Quando si eseguiva il retraining escludendo i dati della classe $c$, i client semplicemente non inviavano gradienti per quella riga del classificatore[cite: 2].
* I pesi di $W^u$ per la classe $c$ **rimanevano congelati allo stato pre-addestrato**, identico a quello di $W^o$[cite: 2].
* Di conseguenza, la divergenza $\Delta w_c = \|w_o[c] - w_u[c]\|_1$ era appiattita su valori prossimi allo zero, facendo fallire l'Algoritmo 3 di FUIA a causa della mancata separazione statistica rispetto alle altre classi[cite: 2].

### B. Perché con FedPrune il Pre-training Funziona Ottimamente ($S_d = 0.7344$)
In FedPrune (Wang et al., 2022), l'unlearning non riparte da zero ma viene applicato **a posteriori** direttamente sul modello addestrato $W^o$[cite: 8].
* Con il pre-training attivo (`PRETRAIN_EPOCHS = 5`), la rete sviluppa filtri convoluzionali e parametri di classificazione solidi per la classe $c$[cite: 1].
* L'algoritmo impone una potatura esplicita e un azzeramento deterministico dei parametri associati: $w_u[c] \leftarrow 0$ e $b_u[c] \leftarrow 0$[cite: 8].
* La divergenza tra il modello originale e quello unlearned diventa massimale[cite: 2, 8]:
  $$\Delta w_c = \|w_o[c] - 0\|_1 = \|w_o[c]\|_1 \gg 0$$
* Poiché nei 3 round di fine-tuning federato le altre 9 classi subiscono variazioni minime, lo scostamento relativo alla classe 3 domina completamente la normalizzazione dello score[cite: 8], registrando un valore di **$S_d[3] = 0.7344$** e un distacco netto rispetto a tutte le classi residue.

---

## 3. Differenze Chiave: Retraining vs. FedPrune

| Proprietà | Retraining da Zero (Exact FU)[cite: 2] | FedPrune / CDP (Wang et al., 2022) |
| :--- | :--- | :--- |
| **Punto di Partenza** | Checkpoint iniziale $W_{\text{init}}$ | Pesi convergenti del modello globale $W^o$[cite: 9] |
| **Round Federati di Unlearning** | 50 – 80 round completi[cite: 1, 2] | **3 round** di recupero rapido[cite: 9] |
| **Tempo di Calcolo Unlearning** | ~180 secondi (3 minuti)[cite: 1] | **7.5 secondi** (riduzione $\approx 96\%$)[cite: 9] |
| **Comportamento Pesi Target** | Mancato aggiornamento (stazionari su $W_{\text{init}}$) | **Azzeramento forzato** ($w_u[c] \leftarrow 0$)[cite: 8] |
| **Compatibilità con Pre-training** | Incompatibile (annulla la divergenza differenziale) | **Compatibile** (massimizza $\Delta w_c$ e amplifica $S_d$) |
| **Vulnerabilità a FUIA ($S_d$)** | Vulnerabile (successo pieno a 1-3 classi)[cite: 5] | **Vulnerabile** (successo pieno con $S_d \ge 0.44$)[cite: 9] |

---

## 4. Considerazioni di Sicurezza (Vulnerabilità a FUIA)

I risultati sperimentali dimostrano due conclusioni primarie:
1. **Efficacia Operativa:** FedPrune riduce drasticamente i costi computazionali e la latenza temporale necessari a rimuovere una classe da un modello federato (da minuti a pochi secondi)[cite: 9].
2. **Persistenza della Vulnerabilità:** L'azzeramento mirato dei parametri del classificatore imposto da FedPrune lascia un'impronta strutturale nei pesi di ampiezza pari all'intera norma $L_1$ del modello originale[cite: 2, 8]. Questo fa sì che l'attacco di discriminazione FUIA (Algoritmo 3) individui la classe rimossa senza ambiguità, dimostrando che **la vulnerabilità di inferenza di FUIA non è limitata al retraining da zero, ma coinvolge anche i meccanismi di approximate unlearning basati su potatura**[cite: 2, 9].

---

## 5. Riferimenti Bibliografici

- J. Wang, S. Guo, X. Xie, and H. Qi, *"Federated Unlearning via Class-Discriminative Pruning"*, in Proceedings of the ACM Web Conference 2022 (WWW '22), pp. 622–632, 2022. https://arxiv.org/abs/2110.11794 (PDF: https://arxiv.org/pdf/2110.11794)
- L. Zhou, Y. Zhu, and R. Liu, *"Model Inversion Attack Against Federated Unlearning"*, IEEE Transactions on Information Forensics and Security, vol. 21, pp. 2342–2357, 2026. https://ieeexplore.ieee.org/document/11400570
- N. Romandini, A. Mora, C. Mazzocca, R. Montanari, and P. Bellavista, *"Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics"*, IEEE Transactions on Neural Networks and Learning Systems, 2024. https://doi.org/10.1109/TNNLS.2024.3478334
- O. Piazzi, *"Analisi e implementazione di attacchi per ricostruzione dati nel Federated Unlearning"*, Relazione di Tirocinio. https://github.com/ottonepiazzi/federated-learning
