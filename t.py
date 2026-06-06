from tqdm import tqdm
import os

# counting files
root = 'data/PET-EARL/domain_chb/'

count = 0
for subject in tqdm(os.listdir(root), position=0, desc='Counting files', leave=True):
    subject_folder = os.path.join(root, subject)
    if os.path.isdir(subject_folder):
        for file in os.listdir(subject_folder):
            if file.startswith('pseudo-earl1'):
                count += 1
print(f'Total number of files starting with the tag: {count}')
