from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json

def main():
    root = Path('/hive/hubmap/lz/tedz-share/HUBMAP_DATA/')
    florida = root / 'Florida'
    stanford = root / 'Stanford'

    lymphnodes = florida / 'LN'
    spleen = florida / 'SPLEEN'
    thymus = florida / 'THYMUS'
    largeint = stanford / 'LI'
    smallint = stanford / 'SI'

    tissue_list = [lymphnodes, spleen, thymus, largeint, smallint]
    file_list = ['lymph nodes', 'spleen', 'thymus', 'large intestine', 'small intestine']
    count = 0

    output_dir = Path.home() / 'analysis'
    tissue_snr = []
    qsl = []

    for i in tissue_list:
        # search for feature files
        # mean_pathlist = list(i.glob(r'**/*_channel_mean.csv'))
        # covar_pathlist = list(i.glob(r'**/*_channel_covar.csv'))
        # total_pathlist = list(i.glob(r'**/*_channel_total.csv'))
        # shape_pathlist = list(i.glob(r'**/*-cell_shape.csv'))
        # meanall_pathlist = list(i.glob(r'**/*_channel_meanAll.csv'))
        qm_pathlist = list(i.glob(r'**/*.json'))
        # texture not included since was turned off for this codex dataset release

        #get qm data
        snr, qs = qm_process(qm_pathlist)
        tissue_snr.append(snr)
        qsl.append(qs)


        # continue
        # seg_list = [mean_pathlist, covar_pathlist, total_pathlist]

        mean_paths = feature_seg(mean_pathlist)
        covar_paths = feature_seg(covar_pathlist)
        total_paths = feature_seg(total_pathlist)

        seg_list = [mean_paths, covar_paths, total_paths]

        c = []
        for j in seg_list:
            a = []
            for k in j:
                f = pd.concat([pd.read_csv(t) for t in k])
                a.append(f)
            a = np.array(a)
            c.append(a)

        mean_matrix = c[0]
        covar_matrix = c[1]
        total_matrix = c[2]
        mean_all_matrix = pd.concat([pd.read_csv(t) for t in meanall_pathlist]).to_numpy()
        shape_matrix = pd.concat([pd.read_csv(t) for t in shape_pathlist]).to_numpy()

        # reformat covar and total
        # covar_matrix = covar_matrix.reshape(
        #     (covar_matrix.shape[0], covar_matrix.shape[1], covar_matrix.shape[2] * covar_matrix.shape[3]))
        covar_matrix = np.concatenate(covar_matrix, axis=1)

        # total_matrix = total_matrix.reshape(
        #     (total_matrix.shape[0], total_matrix.shape[1], total_matrix[2] * total_matrix[3]))
        total_matrix = np.concatenate(total_matrix, axis=1)

        perplex = 35
        tsne_header = [str(1) + "st PC", str(2) + "nd PC"]
        n_iter = 1000
        lr = 1

        full_matrix = np.concatenate((covar_matrix, total_matrix, mean_all_matrix, shape_matrix), axis=1)
        ee = len(full_matrix) / 10

        #prune for infs and nans
        df = pd.DataFrame(full_matrix)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        #full_matrix = df.to_numpy()

        #save csv of the full matrix
        out = output_dir / (file_list[count] + '_full.zip')
        df.to_csv(out, index=False)


        full_matrix = df.to_numpy()

        full_matrix_og = full_matrix.copy()
        tries = 0
        while True:
            try:
                m = PCA(n_components=2, svd_solver="full").fit(full_matrix)
                break
            except Exception as e:
                print(e)
                print("Exceptions caught: ", tries)
                if tries == 0:
                    m = PCA(n_components=2, svd_solver="randomized").fit(full_matrix)
                    tries += 1
                else:
                    print("halving the features in tSNE for PCA fit...")
                    n_samples = int(full_matrix.shape[0] / 2)
                    idx = np.random.choice(
                        full_matrix_og.shape[0], n_samples, replace=False
                    )
                    full_matrix = full_matrix_og[idx, :]

        full_matrix = m.transform(full_matrix_og)

        full_matrix_tsne = full_matrix.copy()
        tsne = TSNE(
            n_components=2,
            perplexity=perplex,
            early_exaggeration=ee,
            learning_rate=lr,
            n_iter=n_iter,
            init='pca',
            random_state=42
        )

        while True:
            try:
                tsne_all = tsne.fit_transform(full_matrix)
                break
            except Exception as e:
                print(e)
                print("halving dataset in tSNE for tSNE fit...")
                n_samples = int(full_matrix.shape[0] / 2)
                idx = np.random.choice(full_matrix_tsne.shape[0], n_samples, replace=False)
                full_matrix = full_matrix_tsne[idx, :]

        #cluster tSNE
        #tsne_cluster = KMeans(n_clusters=3, random_state=42).fit(tsne_all)
        #tsne_labels = tsne_cluster.labels_

        df = pd.DataFrame(tsne_all, index=list(range(1, tsne_all.shape[0] + 1)))
        f = output_dir / (file_list[count] + '-tsne.csv')
        df.to_csv(f)

        #f2 = f.parent / (file_list[count] + '-tsne_plot.pdf')
        #plt.scatter(tsne_all[:, 0], tsne_all[:, 1], c=tsne_labels, marker='.')
        #plt.savefig(f2, format='pdf')

        count += 1

    #snr from all tissues
    plt.hist(tissue_snr, label=file_list)
    plt.legend()
    plt.title('Signal to Noise Ratio of Common Channels Among Different Tissues')
    plt.ylabel('Frequency')
    plt.xlabel('Average Signal to Noise')
    plt.clf()


def qm_process(paths):
    dict_l = []
    qsl = []
    for i in paths:
        qm = json.load(i.read_bytes())
        s2n_otsu = qm['Image Quality Metrics not requiring image segmentation']['Signal To Noise Otsu']
        s2n_z = qm['Image Quality Metrics not requiring image segmentation']['Signal To Noise Z-Score']
        qs = qm['Segmentation Evaluation Metrics']['QualityScore']

        #common channel names
        common_otsu = [s2n_otsu['CD11c'], s2n_otsu['CD21'], s2n_otsu['CD4'], s2n_otsu['CD8'], s2n_otsu['Ki67']]
        common_z = [s2n_z['CD11c'], s2n_z['CD21'], s2n_z['CD4'], s2n_z['CD8'], s2n_z['Ki67']]
        a = np.asarray(common_otsu)
        b = np.asarray(common_z)

        # c = np.sum((a + b)) / len(a)
        c = np.stack((a, b))
        dict_l.append(c)
        qsl.append(qs)

    return dict_l, qsl


def prune_matrix(full_matrix):
    df = pd.DataFrame(full_matrix)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    full_matrix = df.to_numpy()

    return full_matrix
def feature_seg(j):
    # filter for segmentation mask
    cell_pathlist = [path for path in j if 'cell_channel' in path.stem]
    nuclei_pathlist = [path for path in j if 'nuclei' in path.stem]
    cellb_pathlist = [path for path in j if 'cell_boundaries' in path.stem]
    nucleib_pathlist = [path for path in j if 'nucleus' in path.stem]

    return [cell_pathlist, nuclei_pathlist, cellb_pathlist, nucleib_pathlist]


if __name__ == "__main__":
    main()
