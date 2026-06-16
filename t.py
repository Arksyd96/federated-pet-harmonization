import os
import glob
import pandas as pd
from tqdm import tqdm

input_dir = 'outputs/pseudo-earl/'

search_pattern = os.path.join(input_dir, "**", "*radiomics.csv")
csv_files = glob.glob(search_pattern, recursive=True)

assert csv_files, "Aucun fichier CSV de radiomiques trouvé dans le répertoire spécifié."
    
print(f"📄 {len(csv_files)} fichiers trouvés. Début de la lecture...")

df_list = []
for file_path in tqdm(csv_files, desc="Lecture ..."):
    df = pd.read_csv(file_path)
    
    if df.empty: 
        continue
    
    parts = os.path.normpath(file_path).split(os.sep)
    
    # On vérifie qu'on a assez de profondeur (au moins 4 éléments : le fichier + 3 dossiers)
    if len(parts) >= 4:
        domain = parts[ -4 ]
        
        # Déduction de la Source et de la Target selon le nom du domaine
        if "_to_" in domain:
            domain_source, domain_target = domain.split("_to_", 1)
        else:
            domain_source = domain
            domain_target = ""
            
        # Insertion des métadonnées contextuelles au début du DataFrame
        df.insert(0, 'Domain_Target', domain_target)
        df.insert(0, 'Domain_Source', domain_source)
        df.insert(0, 'Domain', domain)
    else:
        df.insert(0, 'Source_Path', os.path.dirname(file_path))
    
    df_list.append(df)
        

assert df_list, "Aucun DataFrame valide n'a été créé à partir des fichiers CSV"

print("🔗 Concaténation des DataFrames...")
master_df = pd.concat(df_list, ignore_index=True)

output_csv_path = os.path.join(input_dir, 'master_radiomics.csv')
master_df.to_csv(output_csv_path, index=False)

print("\n" + "="*50)
print("✅ FUSION TERMINÉE AVEC SUCCÈS")
print("="*50)
print(f"Total des lignes  : {len(master_df)}")
print(f"Total des colonnes: {len(master_df.columns)}")
print("="*50)

master_df.head()
