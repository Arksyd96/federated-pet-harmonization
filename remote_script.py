import argparse
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Argument Parsing
parser = argparse.ArgumentParser(description="Analyse des maxima SUV dans des volumes PET NIfTI.")
parser.add_argument("--data_dir", type=str, required=True, help="Répertoire racine contenant les données PET NIfTI.")
args = parser.parse_args()

# --- CONFIGURATION ---
ROOT_DIR = args.data_dir  # Répertoire racine contenant les données PET NIfTI

pet_max_values = []

print(f"Exploration de {ROOT_DIR} ...")

# 1. Récupération des fichiers
pet_files = []
for root, dirs, files in os.walk(ROOT_DIR):
    for filename in files:
        if filename.startswith("PET_") and filename.endswith(".nii.gz"):
            pet_files.append(os.path.join(root, filename))

print(f"Analyse des maxima pour {len(pet_files)} patients...")

# 2. Boucle rapide : Juste le max par volume
for file_path in tqdm(pet_files):
    try:
        # On charge juste le header si possible pour vérifier, mais nibabel charge tout en lazy loading
        # get_fdata() charge en mémoire, on calcule le max et on libère
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # On nettoie les valeurs aberrantes négatives ou NaN
        valid_max = np.nanmax(data)
        
        # Sécurité : Si le max est infini (bug d'acquisition), on ignore
        if np.isfinite(valid_max):
            pet_max_values.append(valid_max)
            
    except Exception as e:
        print(f"Erreur lecture {file_path}: {e}")

# 3. Analyse Statistique Détaillée
if not pet_max_values:
    print("Aucune donnée trouvée.")
    exit()

pet_max_values = np.array(pet_max_values)
total_patients = len(pet_max_values)

# Calcul des percentiles
percentiles_to_check = [90, 95, 98, 99, 99.5, 99.9]
stats = {}

for p in percentiles_to_check:
    thresh = np.percentile(pet_max_values, p)
    # Nombre de patients qui dépassent ce seuil (qui seront clippés)
    n_clipped = np.sum(pet_max_values > thresh)
    n_kept = total_patients - n_clipped
    stats[p] = (thresh, n_kept, n_clipped)

absolute_max = np.max(pet_max_values)

print("\n" + "="*60)
print(f"DISTRIBUTION DES MAXIMA SUV (Sur {total_patients} patients)")
print("="*60)
print(f"{'Percentile':<10} | {'Seuil (SUV)':<12} | {'Patients OK':<12} | {'Patients Clippés':<15}")
print("-" * 60)

for p in percentiles_to_check:
    thresh, kept, clipped = stats[p]
    print(f"P{p:<9} | {thresh:<12.2f} | {kept:<12} | {clipped:<15} ({100*clipped/total_patients:.1f}%)")

print("-" * 60)
print(f"Max Absolu : {absolute_max:.2f} SUV (Patient le plus chaud ou artefact)")
print("="*60)

# Interprétation automatique
recommended_p = 98 # Par défaut
for p in percentiles_to_check:
    thresh, kept, clipped = stats[p]
    # Règle empirique : on tolère de clipper ~2-5% des patients si ça permet de gagner beaucoup en dynamique
    if clipped <= (0.05 * total_patients): 
        recommended_p = p
        break

rec_thresh, _, rec_clipped = stats[recommended_p]
print(f"\n💡 RECOMMANDATION : Utilise le P{recommended_p} (Seuil = {rec_thresh:.2f})")
print(f"   Cela permet de couvrir {total_patients - rec_clipped} patients sans aucune perte,")
print(f"   et de ne saturer légèrement que les {rec_clipped} patients les plus extrêmes.")