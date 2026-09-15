# Journal de Développement — Harmonisation TEP par Unlearning VAE
**Dernière mise à jour** : Septembre 2026

Ce fichier sert de **mémoire persistante** entre les sessions de travail (changements de poste, changements de modèle IA). Toute nouvelle observation, tentative ou conclusion doit être **ajoutée à la fin** sans supprimer les sections existantes.

---

## 0. Contexte du Projet

### A. Objectif
Ce projet vise à **harmoniser des images TEP (PET)** issues de **5 centres hospitaliers différents**. Chaque scanner TEP imprime une signature subtile sur les images qu'il produit (bruit statistique, calibration, textures liées au détecteur). Cette signature n'a rien à voir avec la biologie ou l'anatomie du patient : c'est un artefact purement machine. L'objectif est de **supprimer cette signature** pour que les images soient comparables en radiomiques entre centres, sans altérer le contenu médical.

### B. Approche : VAE Désentrelacé + Unlearning de Dinsdale
L'architecture repose sur un **Variational Autoencoder (VAE)** dont l'encodeur est bifurqué (`BifurcatedContentStyleEncoder`) :
- **Branche Content** : Encode l'information anatomique/biologique utile dans un tenseur spatial (`z_content`, de dimension `latent_channels × D' × H' × W'`). Ce tenseur est régularisé par une KL divergence spatiale.
- **Branche Style** : Encode la signature du scanner dans un vecteur 1D (`z_style`, de dimension `style_channels`). Ce vecteur est régularisé par une KL divergence 1D, ce qui force l'espace latent du style à être centré sur **zéro** (distribution $\mathcal{N}(0, 1)$).

Le décodeur (`StyleConditionedDecoder`) reconstruit l'image à partir de `z_content`, conditionné par `z_style` via des blocs **AdaIN (Adaptive Instance Normalization)**.

**À l'inférence**, on met `z_style = 0` (le centre de la gaussienne, i.e. un "style neutre/moyen") pour produire une image dépourvue de signature scanner. L'idée est que si le style est bien isolé dans `z_style`, l'image reconstruite sans style sera harmonisée.

### C. Le Pipeline d'Unlearning (Méthode de Dinsdale)
L'entraînement se déroule en deux phases dans le module `UnlearningVAE` (PyTorch Lightning) :

**Stage 1 — Warmup** : Le VAE apprend à reconstruire les images (Loss L1 + SSIM + KL). Simultanément, deux classifieurs de domaine apprennent à identifier le centre hospitalier :
- `style_classifier` : MLP qui classifie `z_style` → doit atteindre une haute accuracy (le style capture bien la signature).
- `content_classifier` : Classifie `z_content` → doit aussi atteindre une haute accuracy pendant le warmup (preuve que la signature est détectable dans le contenu).

**Stage 2 — Unlearning** : Trois étapes adversariales alternées par batch :
- **Étape A** : Le VAE complet est optimisé pour la reconstruction (L1 + SSIM + KL).
- **Étape B** : Les classifieurs s'entraînent sur des tenseurs détachés (`detach()`) pour rester des experts de la signature.
- **Étape C** : L'encodeur seul est optimisé via une **confusion loss** (KL vers distribution uniforme) sur `z_content`. Le but est de forcer `z_content` à devenir **invariant au centre** : le classifieur ne doit plus pouvoir deviner le centre à partir du contenu. La signature doit migrer intégralement dans `z_style`.

### D. Le Rôle Critique de la FFT
La signature du scanner est un bruit très latent, souvent caché dans les **hautes fréquences spatiales** de l'image. Sans aide, le réseau peine à isoler ce signal parmi toute l'information anatomique.

Pour cela, une couche `LearnableFFTHighPassFilter` est appliquée en entrée. Elle extrait une image filtrée passe-haut (les contours et textures haute fréquence) qui est **concaténée** à l'image originale avant d'entrer dans le tronc commun de l'encodeur :
```
input = concat(X, FFT_highpass(X))   →   input_conv   →   branches content/style
```
Sans cette FFT, la classification du centre échoue quasi systématiquement. Avec, on a pu atteindre ~80% d'accuracy.

**Attention** : La FFT crée un risque de triche. Si la confusion loss (Étape C) est autorisée à remonter ses gradients jusqu'au filtre FFT, le réseau peut "tricher" en modifiant les poids du filtre pour masquer les fréquences discriminantes, au lieu de réellement nettoyer le tenseur de contenu. C'est pourquoi un **Stop-Gradient** sur la FFT et `input_conv` pendant l'Étape C a été envisagé (voir Section 3).

### E. Le Défi Actuel : La Classification du Centre
Le problème fondamental n'est ni la reconstruction ni le désapprentissage en soi. **C'est la capacité des classifieurs à détecter la signature du scanner.** Si les classifieurs n'arrivent pas à identifier le centre de manière fiable pendant le Warmup, la confusion loss du Stage 2 est inutile (on ne peut pas désapprendre ce qu'on n'a jamais appris).

Observations clés :
- Le `style_classifier` (MLP sur `z_style`) arrive à classifier raisonnablement bien (~80% acc quand `latent_channels=256`).
- Le `content_classifier` (actuellement un `SpatialDomainClassifier` avec MaxPool + MLP sur `z_content`) a beaucoup plus de mal. C'est sur ce point que se concentrent les efforts.
- Une fois que les deux classifieurs seront robustes, le mécanisme d'unlearning (confusion loss) fonctionnera naturellement — la mécanique adversariale de Dinsdale est éprouvée.

**Prochaine étape décidée** : Entraîner un **classifieur CNN/ResNet indépendant** (sans VAE, sans reconstruction) pour valider formellement quelle architecture, quel pooling, et quelles transformations (FFT, Instance Norm) permettent d'isoler la signature du scanner. Une fois cette base prouvée, on réintégrera le bloc dans le pipeline d'unlearning.

---

## 1. Correction du Dropout de Style (CFG vs Regularization)
Le dropout initial détruisait l'encodeur de style pendant le Warmup (Stage 1).
- **Le Bug :** Le masque de dropout (mise à zéro du vecteur style) était appliqué *avant* le passage dans le `style_classifier`. Le classifieur tentait de deviner le scanner à partir d'un vecteur nul 10% du temps, générant une Cross-Entropy énorme qui ruinait l'encodeur.
- **La Solution (Dropout Hybride) :** Le code applique désormais deux masques **uniquement pour le décodeur** (`z_style_dec = mask_cfg * mask_local * z_style`).
  - `mask_cfg` : Dropout complet du vecteur (10%) pour simuler le Classifier-Free Guidance (CFG) et forcer l'harmonisation.
  - `mask_local` : Dropout élémentaire par neurone (10%) pour régulariser l'espace latent.
- L'inférence utilise un `alpha` (ex: 0.1) pour harmoniser tout en gardant un grain naturel (bruit TEP réaliste).

## 2. L'Explosion de la KLD et l'Anti-Triche (Content Norm)
Lors du passage au Stage 2 (Unlearning), une explosion massive de la `loss_kl_content` (jusqu'à $10^9$) a été observée à l'époque 39.
- **L'Analyse :** Le réseau "trichait" pour satisfaire la *confusion loss* en décalant la moyenne de son encodage vers l'infini (mean-shift). Le réseau savait que la normalisation d'instance (IN) du décodeur allait recentrer le tenseur de toute façon.
- **La Solution (`content_norm`) :** La fonction `forward` du VAE renvoie désormais directement le tenseur normalisé : `z_content_norm = self.vae.content_norm(z_content)`.
- **Conséquence :** Le `content_classifier` lit *exactement* le même tenseur normalisé que le décodeur. La triche du mean-shift est mathématiquement bloquée.

## 3. Le Tronc Semi-Perméable (Stop-Gradient Directionnel)
Dans l'architecture bifurquée, un "Tug-of-War" (conflit de gradients destructeur) se produisait sur le tronc commun (`fft_filter` et `input_conv`) :
- L'Étape A (VAE) tirait sur le tronc pour *préserver* la signature machine (pour la reconstruction).
- L'Étape C (Confusion) tirait sur le tronc pour *détruire* cette signature.
- **La Solution (État de l'art) :** Plutôt que de figer complètement le tronc (ce qui bloque l'amélioration globale) ou de séparer les encodeurs (ce qui cause du *Shortcut Learning*), nous avons transformé le tronc en **valve unidirectionnelle**.
  - **Étape A (Reconstruction)** : Le tronc entier est libre (`requires_grad = True`). Il s'affine et apprend de meilleurs filtres.
  - **Étape C (Désapprentissage)** : Un *Stop-Gradient* est appliqué sur le tronc (`requires_grad = False` sur FFT et `input_conv`). La *confusion loss* ne peut remonter que dans les couches profondes exclusives au contenu.

## 4. Remplacement de l'Average Pooling par le Max Pooling
- Le `AdaptiveAvgPool3d` lissait excessivement les signaux avant classification.
- **Modification :** Remplacement par `AdaptiveMaxPool3d` dans l'encodeur de style (`BifurcatedContentStyleEncoder`) et dans le `SpatialDomainClassifier`.
- **Raison :** Le bruit de scanner (signature de la machine TEP) est composé de pics de hautes fréquences. Le Max Pooling permet au réseau de repérer directement ces pics de bruit plutôt que de les diluer dans une moyenne spatiale.

## 5. Ajustements Techniques et d'Équilibrage
- **Composite Score (Stage 1) :** Corrigé pour récompenser une haute "Accuracy" du classifieur de contenu pendant le Warmup, afin de ne pas fausser le monitoring et l'Early Stopping.
- **Boucle du Classifieur Style (`k_style_steps = 2`) :** Validée. Le classifieur de style s'optimise deux fois par batch (sur des tenseurs détachés) pour rester un "expert" absolu, sans ralentir l'entraînement (coût temporel nul pour 300k paramètres).
- **Optimisation Mémoire :** Ajout global de `set_to_none=True` dans les `zero_grad()` pour accélérer les passes backward.
- **Logging :** La `loss_kl_style` est désormais loggée correctement pendant le Train (Stage 1 et Stage 2) et la Validation.
- **DataLoader :** Bien que les optimisations classiques (`persistent_workers=True`) soient recommandées pour Windows, le `MultiDomainUnlearningDataModule` utilise `tio.Queue` de TorchIO, qui spawn des threads natifs. PyTorch ne doit donc PAS utiliser ses propres workers (`num_workers=0` requis pour le DataLoader).

## 6. Chronologie et Analyse Physique de la Signature Machine (Septembre 2026)

*Note : Cette section a été entièrement réécrite pour refléter fidèlement l'historique de nos expérimentations, y compris les erreurs de raisonnement et les impasses architecturales causées par des recommandations contradictoires.*

### A. La Nature Latente de la Signature
La signature de la machine (scanner TEP) n'est pas un trait sémantique évident (comme l'anatomie ou la biologie du patient). C'est un bruit de fond, une calibration ou une texture très latente, souvent cachée dans les hautes fréquences. 
- L'utilisation d'une **couche FFT en entrée** semble indispensable pour aider le réseau à isoler ce signal. Sans elle, la classification échoue. 
- *Alerte sur la Confusion* : Si une gamme de fréquences précise permet de trouver la machine, il faut s'assurer que l'unlearning ne se contente pas d'effacer les contours FFT concaténés, mais nettoie bien l'image `X` elle-même. D'où l'idée d'un **Stop-Gradient** sur la FFT (pour la figer comme simple extracteur) afin d'obliger la confusion à agir sur les couches intermédiaires.

### B. Le Goulot d'Étranglement (Capacité de l'Espace Latent)
- Les petits espaces latents (`latent_channels` = 8 ou 16) ou même équivalents au volume de l'input (`128 * 8 * 8 * 8 = 65536`) détruisent la signature scanner lors de la compression.
- **Le succès des 256 canaux :** Dans une version antérieure du code (`2026_09_10_183923`), augmenter la capacité à `latent_channels = 256` a permis d'atteindre une accuracy de classification d'environ 80%. Cela prouve que le signal est très fin et nécessite une énorme bande passante pour survivre à la compression.

### C. Le Contresens Mathématique (Instance Norm + Pooling)
C'est ici que l'architecture est partie dans tous les sens suite à des déductions erronées de l'IA. Reprenons l'historique :
1. **L'Origine (Code 2D) :** Dans la toute première version 2D du code, une `InstanceNorm2d` était appliquée *avant* de retourner `mu_content` et `logvar_content`. Pour classifier ce tenseur, on utilisait un `AdaptiveMaxPool2d(1)`.
2. **Le Bug du Zéro Absolu :** En passant à la 3D, l'IA a conseillé de garder l'`InstanceNorm` avant le classifieur, tout en recommandant de remplacer le Max Pooling par un `AdaptiveAvgPool3d` (ou PatchGAN) car le MaxPool détruit la statistique globale du bruit. 
3. **Le Paradoxe :** Appliquer une `InstanceNorm` force la moyenne spatiale de chaque canal à être exactement **Zéro**. Si le classifieur utilise ensuite un Average Pooling sur ce tenseur, il reçoit un vecteur composé mathématiquement **uniquement de zéros** ! Le modèle devenait instantanément aveugle.
4. **Conclusion :** Soit on classifie le vecteur *avant* l'application de l'Instance Norm (comme l'intuition initiale de l'utilisateur le suggérait), soit on ne peut absolument pas utiliser d'Average Pooling après une Instance Norm.

### D. Prochaine Étape Scientifique : L'Isolation Pure
Pour arrêter de tourner en boucle sur des suppositions architecturales contradictoires (MaxPool vs AvgPool vs PatchGAN, InstanceNorm avant/après), la décision est prise de **séparer le problème**.
- Nous allons créer un **Classifieur Indépendant (CNN/ResNet)** sans aucune mécanique de VAE ni de reconstruction.
- Objectif : Valider formellement quelle architecture, quel pooling, et quelle transformation (FFT, Instance Norm) permet d'isoler la signature du scanner à 100%. 
- Une fois cette base de classification solide et prouvée, on réintégrera le bloc vainqueur dans l'architecture complexe de l'Unlearning VAE.

---

## 7. Analyse Critique des Choix Architecturaux (15 Sept. 2026)

### A. L'Instance Norm : Un Dilemme Sans Réponse Simple
Le `content_norm` (InstanceNorm) a été introduit pour bloquer la triche du *mean-shift* (l'encodeur qui pousse les moyennes spatiales vers l'infini pour tromper le classifieur, sachant que le décodeur recentre tout). Cette explosion de KLD ($10^9$) a été observée en pratique.

Cependant, l'Instance Norm supprime la **moyenne et la variance par canal** — or la signature d'un scanner TEP est très probablement liée en partie à des différences de calibration d'intensité globale. En l'appliquant avant le classifieur, on détruit potentiellement l'information la plus discriminante.

**Deux problèmes contradictoires :**
- Sans `content_norm` avant le classifieur → l'encodeur triche par mean-shift → explosion KLD
- Avec `content_norm` avant le classifieur → la signature de calibration est effacée → le classifieur est aveugle

**Piste non testée :** Classifier `z_content` **avant** l'Instance Norm, mais garder l'IN pour le décodeur. Stabiliser autrement (KL plus forte sur le contenu, gradient penalty, spectral norm sur l'encodeur).

### B. Pourquoi le `latent_channels = 256` est Nécessaire dans le VAE (mais Probablement Pas dans un Classifieur Isolé)
Dans le VAE, l'encodeur doit satisfaire **3 objectifs concurrents** qui se disputent la bande passante du latent :
1. **Reconstruction** (L1 + SSIM) : Monopolise la majorité de la capacité pour préserver l'anatomie.
2. **KL Divergence** : Pousse les features vers $\mathcal{N}(0, 1)$, détruisant de l'information.
3. **Classification** : A besoin de features discriminantes du scanner — elle récupère les miettes.

Avec 8 canaux, le budget est trop serré : la reconstruction mange tout et la signature est sacrifiée. Avec 256, il y a enfin assez de marge pour que le signal survive.

Dans un classifieur indépendant, **100% de la capacité** est dédiée à la discrimination du centre. On s'attend à obtenir de bons résultats avec un réseau beaucoup plus compact, ce qui confirmera que le problème est la compétition dans le VAE, pas la détectabilité intrinsèque du signal.

### C. Le Stop-Gradient : FFT vs `input_conv`
- **Figer la FFT pendant l'Étape C** : Raisonnable. La FFT doit rester un extracteur physique passif. La confusion ne doit pas pouvoir masquer les fréquences discriminantes en modifiant le filtre. Voire rendre la FFT non-apprenante (`register_buffer` au lieu de `nn.Parameter`) mérite d'être testé.
- **Figer `input_conv` pendant l'Étape C** : Probablement **trop agressif**. C'est la couche qui mélange `X` et `FFT(X)`. Si la signature est encodée dès cette couche dans les feature maps, la confusion ne peut plus l'atteindre. La confusion a besoin d'un chemin de gradient suffisamment profond pour agir.

### D. Le `SpatialDomainClassifier` n'est Pas Spatial
Le classifieur actuel (`SpatialDomainClassifier`) fait `AdaptiveMaxPool3d(1) → Flatten → MLP`. C'est un classifieur **global** : il réduit tout le tenseur à un seul scalaire par canal, puis classifie. Malgré son nom, il n'analyse aucun motif spatial.

L'ancien PatchGAN (purement convolutif, sans pooling global) forçait chaque voxel à être invariant au domaine. Il a été abandonné pour des raisons qui n'ont pas été documentées. C'est un candidat à retester lors de l'expérience de classification isolée.

### E. Le Style Dropout Pendant l'Entraînement
Le `style_dropout_p = 0.10` en Stage 2 (actuellement actif) reste problématique. Quand `z_style` est mis à zéro et que la Loss de Reconstruction exige la reconstruction de l'image source (qui contient la signature), l'encodeur est **contraint** d'encoder la signature dans `z_content` pour minimiser la MAE. Cela sabote directement l'objectif de l'unlearning.

**Statut :** Le code conserve le mécanisme de dropout (pour flexibilité), mais la valeur devrait être 0.0 pendant l'entraînement si l'on veut un vrai *disentanglement*.

---

## 8. Résultat Expérimental — Classifieur Isolé (15 Sept. 2026, 15h53)

### Expérience
Un `PatchResNet3D` indépendant (7M de paramètres, 4 ResBlocks avec strides anisotropiques, BatchNorm3d, AdaptiveAvgPool3d → MLP) a été entraîné à classifier les 5 centres directement à partir des patchs TEP bruts, **sans aucune FFT**.

### Résultat
- **CE Loss** : ~0.4 en quelques epochs (~1000 itérations)
- **Accuracy** : 70-80% et en progression
- La signature du scanner est donc **détectable directement** sur l'image brute avec un réseau dédié.

### Conclusion Fondamentale
Le problème n'est **pas** dans la détectabilité du signal. La signature est suffisamment présente pour qu'un CNN la capture sans aide fréquentielle. Le problème est **entièrement interne** à l'architecture du VAE : le chemin que `z_content` parcourt avant d'atteindre le `content_classifier` détruit le signal à travers 3 couches successives :
1. **Compression** : `content_head` projette de 512 → `latent_channels` canaux, sacrifiant l'information non essentielle à la reconstruction.
2. **Bruit de reparamétrisation** : `z = mu + eps * sigma` ajoute du bruit qui noie un signal déjà très subtil.
3. **Instance Norm** : `content_norm` efface la moyenne et la variance par canal — exactement les statistiques qui portent la calibration du scanner.

De plus, la reconstruction (L1 + SSIM) monopolise la bande passante de l'encodeur pour l'anatomie, et la KL pousse tout vers N(0,1). Le classifieur récupère les miettes d'un signal déjà écrasé.

### 5 Mesures Correctives par Ordre de Priorité

**1. Classifier `mu_c` AVANT l'Instance Norm (priorité absolue)**
Le classifieur content doit recevoir `mu_c` brut (avant `content_norm`). L'IN est conservée uniquement pour le décodeur. La triche par mean-shift est contrôlée autrement (voir mesure 4).

**2. Classifier `mu_c` (déterministe) au lieu de `z_content` (reparamétrisé)**
Le bruit epsilon de la reparamétrisation noie la signature fine du scanner. Le classifieur devrait lire le mode de la distribution postérieure (`mu_c`), pas l'échantillon bruité.

**3. Remplacer le MaxPool3d(1) → MLP par un petit CNN convolutif**
Le classifieur actuel fait un MaxPool global qui réduit tout le tenseur spatial à un scalaire par canal. La signature est probablement dans les textures spatiales, pas dans un seul pic. Un petit CNN (2-3 blocs Conv3d+BN) serait plus fidèle à ce que le ResNet a prouvé.

**4. Contraindre le mean-shift autrement que par l'Instance Norm**
Si on retire l'IN du chemin du classifieur, il faut empêcher la triche par mean-shift :
- Augmenter `kl_content_weight` (la KL pénalise directement les moyennes qui dérivent)
- Ajouter une pénalité L2 sur `mu_c.mean(dim=[2,3,4])` (la moyenne spatiale globale)
- Appliquer du spectral normalization sur les couches de l'encodeur

**5. Tester BatchNorm3d vs GroupNorm dans la branche content de l'encodeur**
Le ResNet qui réussit utilise `BatchNorm3d`. L'encodeur VAE utilise `GroupNorm(32)`. Changer la normalisation de l'encodeur impacte tout (reconstruction, KL, style), mais c'est un différenciateur direct entre les deux architectures. À tester en dernier car c'est le changement le plus invasif.