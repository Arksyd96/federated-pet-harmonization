# Mises à Jour de l'Architecture Unlearning VAE (Dernière Itération)
**Date** : Septembre 2026 (Mise à jour suite aux instabilités de KLD et de classification)

Ce document trace les dernières évolutions architecturales critiques implémentées pour stabiliser le VAE désentrelacé (Disentangled VAE) et corriger les explosions de gradients observées lors du Stage 2 (Unlearning).

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