# Contexte et Mise à Jour de l'Architecture Unlearning VAE
**Date** : Septembre 2026
**Objectif** : Harmonisation d'images TEP multicentriques par désapprentissage (Unlearning) de la signature machine (méthode de Dinsdale).

## 1. Le Problème Initial
Malgré l'implémentation de l'architecture *Disentangled VAE* (avec un encodeur bifurqué et un décodeur conditionné par AdaIN), le modèle souffrait d'un problème majeur à l'inférence : **le retrait du vecteur style (`z_style = 0`) ne retirait pas la signature du scanner**. 
Le décodeur reconstruisait parfaitement l'image source, ce qui signifiait que l'encodeur avait contourné la *confusion loss* et encodé la signature machine directement dans le tenseur de contenu (`z_content`). Ce phénomène est connu sous le nom de *Posterior Collapse* ou *Style Bypassing*.

## 2. Analyse des 4 Failles Architecturales

1. **La Faille du Pooling (Classifieur de domaine aveugle)** : 
   La *confusion loss* était appliquée sur `s_content`, un vecteur obtenu après un *Global Average Pooling* spatial (`AdaptiveAvgPool3d`). Le réseau contournait cette perte en cachant la signature machine dans les hautes fréquences et textures (dont la moyenne spatiale s'annule). Le classifieur était trompé, mais le décodeur, lui, récupérait toute l'information texturée via `z_content`.
2. **Absence de KLD sur le Style (Vecteur nul hors distribution)** : 
   La branche style n'avait plus de Kullback-Leibler Divergence (KLD). L'espace latent du style n'était donc pas centré sur $0$. En conséquence, forcer `z_style = 0` à l'inférence revenait à injecter un vecteur aléatoire (hors distribution) que le décodeur ignorait simplement.
3. **Le Style Dropout Contradictoire** : 
   L'entraînement forçait le retrait du style avec `style_dropout_p = 0.05`. Cependant, la loss de reconstruction (`L1`) exigeait toujours la reconstruction parfaite de l'image source (avec sa signature). Cela pénalisait mathématiquement le *disentanglement* et forçait le réseau à sauvegarder la signature dans `z_content`.
4. **Initialisation de l'AdaIN à Zéro** : 
   Les projections de l'AdaIN étaient initialisées à $0$ (`gamma=0, beta=0`). Le vecteur style était donc totalement inactif lors des premières époques. Avec un `z_content` de forte capacité (4096 dimensions), le réseau prenait l'habitude de tout reconstruire sans jamais faire l'effort d'utiliser le canal style.

---

## 3. Modifications Apportées au Code (`harmonization_vae.py`)

Pour forcer un véritable *disentanglement*, les actions suivantes ont été implémentées :

### A. Classifieur de Domaine Spatial (PatchGAN)
- Remplacement du MLP linéaire par un `SpatialDomainClassifier` (CNN 3D).
- Le classifieur prend désormais en entrée le **goulot d'étranglement exact** qui part au décodeur (`z_content`, 8 canaux). La *confusion loss* (Cross-Entropy) est calculée voxel par voxel (`F.cross_entropy` sur les dimensions spatiales), empêchant toute triche via les textures.

### B. Restauration de la KLD du Style
- L'encodeur `style_head` génère désormais `2 * style_channels`.
- La reparamétrisation (`mu_s`, `logvar_s`) a été rétablie.
- L'ajout de `kl_style_weight` dans `UnlearningVAE` force l'espace de style à être une Gaussienne $\mathcal{N}(0, 1)$. À l'inférence, `0` représente désormais mathématiquement un style neutre/moyen.

### C. Corrections de l'Entraînement et de l'AdaIN
- **AdaIN** : Les poids de projection sont désormais initialisés via une loi normale (`std=0.02`), forçant le réseau à gérer le signal de style dès l'époque 1.
- **Dropout** : Le `style_dropout_p` est désactivé (`0.0`) pendant l'entraînement. Le réseau doit utiliser le style de la source pour reconstruire la source.

### D. Améliorations de l'Inférence et du Logging
- La fonction `_log_images` génère désormais **3 images** par validation : `Source`, `Reconstruction (avec style)`, et `Harmonisée (avec style nul)`.
- La fonction `.harmonize()` accepte un paramètre `alpha_style` pour doser l'intensité du style à l'inférence.

---

## 4. Traçabilité des Entraînements
Pour assurer la reproductibilité face aux nombreux changements d'architecture, le script `train_unlearning_harmonization_vae.py` a été modifié.
À chaque lancement de modèle, un dossier `code_snapshot/` est automatiquement généré dans le dossier de log (`runs/...`). Il contient une copie de sauvegarde exacte de :
1. `train_unlearning_harmonization_vae.py`
2. `harmonization_vae.py`
3. `data.py` (DataModule)

