# Journal de Développement — Harmonisation TEP par Unlearning VAE
**Dernière mise à jour** : 18 Novembre 2026

Ce fichier sert de **mémoire persistante** entre les sessions de travail (changements de poste, changements de modèle IA). Toute nouvelle observation, tentative ou conclusion doit être **ajoutée à la fin** sans supprimer les sections récentes ni les acquis architecturaux.

---

## 0. Contexte & Architecture Active

### A. Objectif
Harmoniser des images TEP (PET) issues de plusieurs centres hospitaliers afin de supprimer la signature propre aux scanners (bruit, reconstruction, textures haute fréquence) tout en préservant l'anatomie et les biomarqueurs quantitatifs (SUV, textures tumorales).
- **Centres d'entraînement (5)** : `a100`, `chb`, `lacassagne`, `nantes`, `rennes`.
- **Cohorte externe exclue de l'entraînement** : `curie` (regroupe des patients multi-centres, utilisée pour la validation externe *zero-shot*).
- **Pistes abandonnées / en pause** : `unet_iffn`, `starganv2`, `pet_earl_translation` (abandonnés) ; `unet_skip` (en pause).

### B. Architecture du Modèle Actif (`UnlearningVAE` — `pet_harmonization/models/unlearning_vae.py`)
- **Encodeur bifurqué (`ContentStyleEncoder`)** : deux branches convolutives **indépendantes dès l'entrée** (`content_*` et `style_*`), `BatchNorm3d`, sans FFT (`use_fft=False`), `latent_channels=64` (`mu_c, logvar_c` spatiaux), `style_channels=256` (`mu_s, logvar_s` 1D après `AdaptiveAvgPool3d(1)`).
- **Goulot & Normalisation** : `content_norm = InstanceNorm3d(64, affine=False)` appliqué avant le décodeur (bloque la triche par *mean-shift* qui faisait exploser la KLD).
- **Décodeur (`StyleConditionedDecoder`)** : blocs résiduels conditionnés par `z_style` via **AdaIN** (`StyleEmbedder`).
- **Pipeline de Dinsdale en 2 phases** :
  - **Stage 1 (Warmup, 40 époques)** : reconstruction (`L1 + 0.1 * SSIM`, poids chauffé linéairement de `0.1` à `1.0`) + KLD (`1e-6`) + entraînement des deux classifieurs de domaine (`content_classifier`, `style_classifier`).
  - **Stage 2 (Unlearning adversarial)** : alternance par batch entre reconstruction + `style_classifier` (Étape A), entraînement des classifieurs sur tenseurs détachés (`k_style_steps=2`, Étape B), et confusion loss sur `z_content` *avant* `content_norm` pour rendre le contenu invariant au centre (Étape C).
  - **Dropout de style (CFG)** en Stage 2 : `cfg_drop_prob=0.10` en mode `"gaussian"` (`z_s = torch.randn_like(z_s)` avec `z_c.detach()`).

---

## 1. Acquis Clés de l'Ablation de Septembre 2026 (Rappel)
- **Détectabilité prouvée** : un `PatchResNet3D` isolé (`scripts/train/train_center_classifier.py`) sépare les centres sans FFT dès les premières itérations (~70–80%+), prouvant que le signal scanner est directement accessible.
- **Causes des échecs initiaux sur le VAE** :
  1. Appliquer `InstanceNorm` puis `AdaptiveAvgPool` produisait un vecteur nul pour le classifieur content → confusion loss résolue en classifiant *avant* `content_norm`.
  2. Un goulot trop étroit (`8–32` canaux) provoquait une compétition destructrice (*gradient starvation*) avec la reconstruction et une instabilité en validation → résolu en élargissant à **`latent_channels=64`**, en séparant les deux branches encodeur et en utilisant `BatchNorm3d`.

---

## 2. Modèle Sandbox du 23 Octobre & Évaluation des Stratégies de Style (Oct. – Nov. 2026)

### A. Checkpoint de Référence & Scripts Actifs
- **Entraînement** : `pet_harmonization/models/unlearning_vae.py` (commit `a53610b`, 23 oct. 2026, dropout Gaussien 10%).
  - Fin Stage 1 : `runs/sandbox-unlearn-vae/2026_10_23_162550/checkpoints/stage1_final_epoch=039-rec=0.1122-style=0.365-content=0.901.ckpt`
  - Fin Stage 2 : `runs/sandbox-unlearn-vae/2026_10_23_162550/checkpoints/2-epoch=127-rec=0.0201-style=0.960-content=0.172.ckpt`
- **Inférence & Extraction de styles** : `scripts/inference/inference_patch_harmonization_sandbox.py` et `scripts/inference/extract_center_styles.py`.
- **Évaluation radiomique** : `notebooks/evaluate_harmonization_radiomics.ipynb` et `notebooks/master_radiomics.csv`.

### B. Comparaison des Modes d'Inférence sur le Foie (`VOI == 'liver'`)
Plusieurs injections de `z_style` ont été testées à l'inférence (50 patients/centre sur les 5 centres vus + `curie` *unseen* = 300 patients) :

1. **Mode `gaussian` (vecteur $\mathcal{N}(0, 1)$ fixe partagé — Meilleur compromis actuel)** :
   - Réduit nettement le biais inter-centres : médiane du $H$ de Kruskal-Wallis sur 208 patients appariés (5 centres) : **`84.6 → 41.2`** (et sur 300 patients / 6 centres : `102.9 → 57.7` ; sur `top20_pet` : `139.2 → 57.6`).
   - Préserve très bien le niveau moyen de SUV du foie (`ratio = 0.998`, corrélation `r_mean = 0.947`).
   - Généralise correctement en *zero-shot* sur `curie` (les points de `curie` se mélangent aux autres centres).
   - Surpasse `EARL1/2` et `ComBat` sur la majorité des features radiomiques dans les comparaisons strictes du notebook.

2. **Styles médians extraits (`style-global-median` et `style-nantes`)** :
   - Mélangent bien les centres entre eux dans l'espace VAE (`H_med ≈ 59` sur 300 patients), mais **décalent fortement les intensités vers le bas** (`firstorder_Mean` sur le foie à `-19.4%` pour `style-nantes` et `-28.2%` pour `style-global-median`).
   - Explication mécanique identifiée : `InstanceNorm3d` sur `z_content` délègue une part de la calibration d'intensité globale à AdaIN (`z_style`), et la médiane coordonnée par coordonnée de `mu_s` sur des milliers de patchs (au lieu d'un échantillon `reparameterize(mu_s, logvar_s)` vu au train) réduit la norme du vecteur de style.
   - Le décalage apparent de `nantes` observé sur une PCA initiale venait du fait que les images `nantes` affichées à droite étaient restées en `PET` brut (hors VAE), alors que le passage dans le VAE déplace globalement tous les centres sur la PCA.

3. **Contrôle négatif `original` (style natif conservé, `alpha = 1.0`) — Preuve contre l'effet purement dégradatif** :
   - Deux effets ont été isolés en comparant `PET Standard` → `VAE Original` → `VAE Gaussien` (sur 300 patients / 6 centres, et 250 / 5 centres entraînés) :
     - *Effet autoencodeur/lissage* : passer dans le VAE, même en gardant le style d'origine, lisse les textures haute fréquence les plus fines (classification LogReg 5 centres sur 93 features : `78.0% → 54.8%`, hasard = `20%`).
     - *Effet réel de `z_style` / unlearning* : avec exactement le **même VAE**, garder le style d'origine maintient un fort biais inter-centres (`H_med = 116.9` sur 5 centres), alors qu'imposer `gaussian` le divise par deux (`H_med = 57.5`) et réduit encore la séparabilité (`54.8% → 40.4%`). L'harmonisation ne vient donc pas uniquement d'un floutage aveugle.

---

## 3. Refactorisation Pipeline Données & Préparation Évaluation Lésions (18 Nov. 2026)

- **Séparation reproductible par manifestes CSV figés (`train` / `test` uniquement)** :
  - Ajout dans `MultiDomainUnlearningDataModule` (`pet_harmonization/data.py`) des paramètres optionnels `manifest_csv`, `active_domains` et `eval_splits` (`("test",)` par défaut, `["train"]` ou `["train", "test"]` possible à l'inférence). `split_config` reste fonctionnel si `manifest_csv=None`.
  - Nouveau script `scripts/utils/preprocessing/generate_dataset_manifest.py` pour générer un manifeste stratifié (`train` fixe incluant un quota contrôlé de patients avec lésions + tout le reste en `test`, `curie` à `0` en `train`).
  - Clarification importante sur les runs passés : `SEED: 101` ne pilotait que les seeds globales (`set_seed`), tandis que `MultiDomainUnlearningDataModule` utilisait sa seed isolée par défaut (`42`). Un manifeste de sauvegarde (`data/manifest_seed42.csv`) reflète ce split historique pour le modèle d'octobre.
- **Extraction VOIs fiabilisée** : `scripts/utils/eval/extract_vois.py` force désormais l'orientation canonique `RAS` sur les masques et images et vérifie la correspondance des grilles avant découpe (corrige les crashs PyRadiomics sur les petites lésions).
- **Évaluation latente prête (en attente)** : `scripts/utils/eval/evaluate_latent_unlearning.py` préparé pour comparer Stage 1 vs Stage 2 directement dans `z_content` / `z_style` sans biais de reconstruction.

---

## 4. Prochains Objectifs

1. **Validation clinique aval sur `curie` (cohorte externe multi-centres)** :
   - Prédire l'ensemble de `curie` en mode `gaussian` (avec le même vecteur fixe), segmenter les lésions (ex. `autoPET3`), extraire `SUVmax` avant vs après harmonisation sur les mêmes masques lésionnels, couper à la médiane et comparer les **courbes de survie (Kaplan-Meier + log-rank)**.
2. **Audit de préservation lésionnelle sur les 4 autres centres** :
   - Ré-extraire les VOIs et radiomiques sur `lesion` (`extract_vois.py` + `extract_radiomics.py`) pour `harmonized-pet-vae-3d-sandbox-gaussian` face à `pet` et `earl1/2` afin de vérifier la fidélité de `SUVmax` / `SUVmean` dans les tumeurs (et tester `sample_posterior=False` à l'inférence si la variance de reparamétrisation dégrade les pics).
3. **Preuve directe de l'unlearning dans l'espace latent (Stage 1 vs Stage 2)** :
   - Exécuter `scripts/utils/eval/evaluate_latent_unlearning.py` sur les checkpoints `epoch=039` vs `epoch=127` (*probing* groupé par patient + PCA/ARI/Kruskal-Wallis sur `z_content_raw`, `z_content_norm` et `z_style`).
4. **Ré-entraînement propre & calibration d'échelle** :
   - Utiliser `./data/manifest.csv` pour les prochains entraînements, et évaluer comment éviter la dépendance d'échelle SUV liée à `InstanceNorm3d` + AdaIN pour permettre aussi une translation fidèle d'un domaine vers un autre.
