# Compute perceptual hashes

from math import ceil
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
import random

def compute_hash(fp):
    try:
        return str(imagehash.phash(Image.open(fp).convert('RGB').resize((128, 128))))
    except:
        return None
    
def show_duplicates(df, max_per_group=5):

    dup_hashes = df[df.duplicated('hash', keep=False)]
    grouped = dup_hashes.groupby('hash')

    for hash_val, group in grouped:
        file_paths = group['file_path'].tolist()[:max_per_group]
        n = len(file_paths)
        fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
        for ax, fp in zip(axes, file_paths):
            ax.imshow(Image.open(fp))
            ax.axis('off')
        plt.suptitle(f"Duplicate group: {hash_val} ({len(group)} images total)")
        plt.show()
        
def show_random_samples(df, label, n=5):
    sample_paths = random.sample(df[df['label'] == label]['file_path'].tolist(), n)
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for ax, img_path in zip(axes, sample_paths):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(f"Random {label.capitalize()} Samples")
    plt.show()