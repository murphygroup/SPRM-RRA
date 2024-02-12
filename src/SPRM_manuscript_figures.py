from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import umap
from skimage import measure
import pickle
from scipy import stats
from itertools import combinations
from scipy.stats import spearmanr

### GLOBAL ###

ROOT = Path('/Users/tedzhang/Desktop/CMU/hubmap/SPRM/MANUSCRIPT/input')
TISSUES = ['lymph_nodes', 'spleen', 'thymus', 'large_intestine', 'small_intestine']
OUTPUT_DIR = Path('/Users/tedzhang/Desktop/CMU/hubmap/SPRM/MANUSCRIPT/output')

#HARD CODE COMMON CHANNELS
COMMON_CHANNELS = ['CD11c', 'CD21', 'CD4', 'CD8', 'Ki67']

# random
rng = np.random.default_rng(42)

##############

def main():





if __name__ == "__main__":
    main()