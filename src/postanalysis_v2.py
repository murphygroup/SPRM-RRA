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



def main():
    # root = Path('/hive/hubmap/lz/tedz-share/HUBMAP_DATA/')
    # root = Path('/home/tedz/Downloads/sprm-analysis/inputs/')
    # root = Path('/Users/tedzhang/Downloads/SPRM-analysis/input_new')
    root = Path('/Users/tedzhang/Desktop/CMU/hubmap/SPRM/MANUSCRIPT/input')

    file_list = ['lymph_nodes', 'spleen', 'thymus', 'large_intestine', 'small_intestine']
    # output_dir = Path.home() / 'analysis' / 'output'
    # output_dir = Path.home() / 'Downloads' / 'sprm-analysis' / 'outputs_new'
    output_dir = Path('/Users/tedzhang/Desktop/CMU/hubmap/SPRM/MANUSCRIPT/output')
    # list inits
    otsu_tissue_snr = []
    otsu_tissue_snr_avg = []
    zscore_tissue_snr = []
    zscore_tissue_snr_avg = []
    common_channels = ['CD11c', 'CD21', 'CD4', 'CD8', 'Ki67']
    common_channels_pixels = ['CD11c', 'CD21', 'CD4', 'CD8', 'Ki67', 'pixels']
    # total_imgs = [19, 16, 11, 13, 13]
    # manually get total number of pixels
    pixels = [12664 * 7491, 9975 * 9489, 9973 * 9492, 10003 * 9529, 9997 * 9521]
    # print["".join(a) for a in combinations(common_channels, 2)]
    # markers = ['.', '+', 'x', '_', '*']

    # for covar common channels
    covar_commonc = []
    a = list(combinations(common_channels, 2))
    for i in a:
        covar_commonc.append(i[0] + ':' + i[1])

    # pca list
    pca_l = []
    # tsne_l = []
    # ccn_l = []
    full_matrix_l = []
    qs_l = []

    # random init
    rng = np.random.default_rng(42)

    names = ['covar', 'total', 'mean-all', 'shape']
    for i in range(len(file_list)):
        # search for feature files
        # covar = np.load((root / (file_list[i] + '-covar.npy')), allow_pickle=True)
        # total = np.load((root / (file_list[i] + '-total.npy')), allow_pickle=True)
        # shapes = np.load((root / (file_list[i] + '-shape.npy')), allow_pickle=True)
        # meanall = np.load((root / (file_list[i] + '-mean-all.npy')), allow_pickle=True)
        covar = pd.read_csv((root / (file_list[i] + '-covar')))
        total = pd.read_csv((root / (file_list[i] + '-total')))
        shapes = pd.read_csv((root / (file_list[i] + '-shape')))
        meanall = pd.read_csv((root / (file_list[i] + '-mean-all')))
        qs = pd.read_pickle(root / (file_list[i] + '-qs.pkl'))
        snr_common = np.load(root / (file_list[i] + '-snr_common.npy'), allow_pickle=True)
        total_int = np.load(root / (file_list[i] + '-total_intensity_common.npy'), allow_pickle=True)
        total_int_cells = pd.read_csv((root / (file_list[i] + '-total_cells')))
        # pixels = pd.read_pickle(root / (file_list[i] + '-pixels.pkl'))
        mean_cells = pd.read_csv((root / (file_list[i] + '-mean_cells')))
        covar_cells = pd.read_csv((root / (file_list[i] + '-covar_cells')))
        cellbg = np.load(root / (file_list[i] + '-cellbg.npy'), allow_pickle=True)

        with open(root / (file_list[i] + '-channel_mean_names.pkl'), 'rb') as f:
            mean_channel_names = pickle.load(f)

        # get common channels
        total = total[common_channels]
        meanall = meanall[common_channels]
        covar = covar[covar_commonc]
        total_cells = total_int_cells[common_channels_pixels]
        # covar_cells = covar_cells[covar_commonc]
        # mean_cells = mean_cells[common_channels]

        # inv_pixels = {v: k for k, v in pixels.items()}

        all_feats = [covar, total, meanall, shapes]
        # all_feats = [covar_cells, total_cells, mean_cells, shapes]
        all_feats = pd.concat(all_feats, axis=1)

        # add id
        all_feats['ID'] = i

        #add new quality metrics
        qs = qs.drop(columns=['1/AvgCVBackground', 'AvgCellBG', 'npSM'])

        #snr common df
        snr_columns = ['Otsu:CD11c', 'Otsu:CD21', 'Otsu:CD4', 'Otsu:CD8', 'Otsu:Ki67', 'M/SD:CD11c', 'M/SD:CD21', 'M/SD:CD4', 'M/SD:CD8', 'M/SD:Ki67']
        snr_common_df = pd.DataFrame(snr_common, columns=snr_columns)

        #total int df
        total_columns = ['TotInt:CD11c', 'TotInt:CD21', 'TotInt:CD4', 'TotInt:CD8', 'TotInt:Ki67']
        total_common_df = pd.DataFrame(total_int, columns=total_columns)

        #cellbg df
        cellbg_columns = ['Cell/BG:CD11c', 'Cell/BG:CD21', 'Cell/BG:CD4', 'Cell/BG:CD8', 'Cell/BG:Ki67']
        cellbg_df = pd.DataFrame(cellbg, columns=cellbg_columns)

        qs_df = pd.concat([qs, snr_common_df, total_common_df, cellbg_df], axis=1)

        #add tissue type
        qs_df['Tissue'] = file_list[i]

        qs_l.append(qs_df)
        full_matrix_l.append(all_feats)

        # get background int for specific images
        # pixels_values = list(pixels.values())

        tot_int_d_pixels = []
        for j in range(len(pixels)):
            # tot_int_d_pixels.append(total_int[j] / pixels_values[j])
            tot_int_d_pixels.append(total_int[j] / pixels[j])

        total_int_avg = np.mean(tot_int_d_pixels, axis=0)
        snr_common_avg = np.mean(snr_common, axis=0)
        tot_img = total_int.shape[0]

        # bgi_otsu = []
        # bgi_zscore = []

        # mean intensity per cell
        # find otsu threshold
        # find cells above threshold and average those

        background_int_otsu = total_int_avg / snr_common_avg[:5]
        background_int_z = total_int_avg / snr_common_avg[5:]

        # background_int_z = total_int_avg / pixels[i] / snr_common_avg[5:]
        # bgi_otsu.append(background_int_otsu)
        # bgi_zscore.append(background_int_z)

        new_num = int(total.shape[0] / tot_img)
        #
        otsu_img = []
        zscore_img = []
        #
        otsu_avgpc = []
        zscore_avgpc = []

        for n in range(tot_img):

            rndperm = rng.permutation(total.shape[0])
            sub_total = total.iloc[rndperm[:new_num], :]
            sub_meanall = meanall.iloc[rndperm[:new_num], :]

            # get number of pixels in cell
            # id = pixels_values[n]
            # idx = total_int_cells.index[total_int_cells['pixels'] == id].tolist()
            # sub_total_cells = total_int_cells[total_int_cells['pixels'] == id]
            # sub_total_cells = total_int_cells.iloc[idx, :]
            # sub_mean_cells = mean_cells.iloc[idx, :]

            num_of_pixels = sub_total / sub_meanall
            # num_of_pixels = sub_total_cells / sub_mean_cells
            #
            #
            #     #change all nans to 0
            #     # prune for infs and nans
            # df = pd.DataFrame(num_of_pixels)
            # df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # df.fillna(0, inplace=True)
            num_of_pixels = num_of_pixels.to_numpy()
            #
            idx_l = []
            # find the common channels
            for j in common_channels:
                indices = [i for i, elem in enumerate(mean_channel_names) if j in elem]
                idx_l.append(indices[0])

            # np_cc = num_of_pixels[:, idx_l]
            # total_common = sub_total[:, idx_l]
            np_cc = num_of_pixels
            total_common = sub_total.to_numpy()
            # total_common = sub_total_cells[common_channels].to_numpy()

            denom1 = background_int_otsu * np_cc
            denom2 = background_int_z * np_cc
            #
            otsu_cells = np.zeros((total_common.shape))
            zscore_cells = np.zeros((total_common.shape))
            for j in range(5):
                otsu_cells[:, j] = total_common[:, j] / denom1[:, j]
                zscore_cells[:, j] = total_common[:, j] / denom2[:, j]
            #
            df = pd.DataFrame(otsu_cells)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            otsu_cells = df.to_numpy()
            #
            df = pd.DataFrame(zscore_cells)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            zscore_cells = df.to_numpy()
            #
            #     #avg snr per cell
            avg_snr_otsu = np.mean(otsu_cells, axis=0)
            avg_snr_zscore = np.mean(zscore_cells, axis=0)
            #
            otsu_avgpc.append(avg_snr_otsu)
            zscore_avgpc.append(avg_snr_zscore)
        #
            abv_otsu = []
            abv_zscore = []
            #find abv avg threshold
            for j in range(5):
                idx = np.where(otsu_cells[:, j] > background_int_otsu[j])
                idx2 = np.where(zscore_cells[:, j] > background_int_z[j])

                m = np.mean(otsu_cells[idx, j])
                m2 = np.mean(zscore_cells[idx2, j])

                abv_otsu.append(m)
                abv_zscore.append(m2)

            otsu_img.append(abv_otsu)
            zscore_img.append(abv_zscore)

        otsu_tissue_snr.append(np.asarray(otsu_img))
        zscore_tissue_snr.append(np.asarray(zscore_img))
        otsu_tissue_snr_avg.append(np.asarray(otsu_avgpc))
        zscore_tissue_snr_avg.append(np.asarray(zscore_avgpc))

    print('out of image loop')

################################################################################


    #sample cells to use from each tissue
    step = 1000
    rndperm = rng.permutation(full_matrix_l[0].shape[0])
    rndperm_sublists = [rndperm[i:i+step] for i in range(0, len(rndperm), step)]
    subsample_matrix_l = [x.loc[idx] for (x, idx) in zip(full_matrix_l, rndperm_sublists)]

    subsample_matrix = pd.concat(subsample_matrix_l, axis=0)


    # pca on full features
    # sample down


    # full_matrix = df_subset.to_numpy()
    full_matrix = subsample_matrix.copy()

    full_matrix_noid = full_matrix.drop(['ID', 'Unnamed: 0'], axis=1)

    # prune for nans and infs
    full_matrix_noid.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_matrix_noid.fillna(0, inplace=True)

    # zscore
    full_matrix_pop = stats.zscore(full_matrix_noid)


    # PCA
    full_matrix_og = full_matrix_pop.copy()
    tries = 0
    while True:
        try:
            m = PCA(n_components=2, svd_solver="full").fit(full_matrix_pop)
            break
        except Exception as e:
            print(e)
            print("Exceptions caught: ", tries)
            if tries == 0:
                m = PCA(n_components=2, svd_solver="randomized").fit(full_matrix_pop)
                tries += 1
            else:
                print("halving the features in tSNE for PCA fit...")
                n_samples = int(full_matrix_pop.shape[0] / 2)
                idx = np.random.choice(
                    full_matrix_og.shape[0], n_samples, replace=False
                )
                full_matrix = full_matrix_og[idx, :]

    full_matrix_pca = m.transform(full_matrix_og)

    # # get 2D PCA
    # pca_l.append(full_matrix)

    # #get kmeans cluster
    # nc, pca_labels = get_silhouette_score(full_matrix, 'PCA-silhouette-scores', output_dir)

    full_matrix_np = full_matrix['ID'].to_numpy()
    # for i in range(len(file_list)):
    #     idx = np.where(full_matrix_np == 5-i)
    #     plt.scatter(full_matrix_pca[idx, 0], full_matrix_pca[idx, 1], label=file_list[i], marker=markers[i], alpha=0.5)
    #     plt.legend()

    # plt.savefig(output_dir / 'PCA.png', bbox_inches='tight')
    # plt.clf()
    # plt.close()

    # tSNE on full features
    full_matrix_tsne = full_matrix_pca.copy()
    tsne = TSNE(
        n_components=2,
        perplexity=100,
        early_exaggeration=12,
        learning_rate=1,
        n_iter=1000,
        init="random",
        random_state=42
    )

    while True:
        try:
            tsne_all = tsne.fit_transform(full_matrix_tsne)
            break
        except Exception as e:
            print(e)
            print("halving dataset in tSNE for tSNE fit...")
            n_samples = int(full_matrix_tsne.shape[0] / 2)
            idx = np.random.choice(full_matrix_tsne.shape[0], n_samples, replace=False)
            full_matrix_tsne = full_matrix_tsne[idx, :]

    #add tsne to full_matrix as a column
    full_matrix['tsne_embedding_x'] = tsne_all[:, 0]
    full_matrix['tsne_embedding_y'] = tsne_all[:, 1]
    #shuffle for randomized plotting
    # tsne_shuffle = full_matrix.sample(frac=1).copy()

    #make tsne and umap figure
    fig, axs = plt.subplots(1, 2)
    # tu = ['A', 'B']

    # for x, y, labels in zip(tsne_shuffle['tsne_embedding_x'], tsne_shuffle['tsne_embedding_y'], tsne_shuffle['ID']):
    for i in range(len(file_list)):
        idx = np.where(full_matrix_np == i)
        axs[0].scatter(tsne_all[idx, 0], tsne_all[idx, 1], label=file_list[i], marker='.', alpha=1)
    axs[0].legend(loc='center', bbox_to_anchor=(0.05, 0.8), bbox_transform=plt.gcf().transFigure)
    axs[0].set_title('A', fontsize=15, fontweight='bold')
    #     axs[0].scatter(x, y, label=labels, alpha=1)
    #     axs[0].legend()

    reducer = umap.UMAP(random_state=42)
    umap_features = full_matrix_pop.copy()
    # umap_features = full_matrix_pca.copy()
    umap_embed = reducer.fit_transform(umap_features)

    full_matrix['umap_embedding_x'] = umap_embed[:, 0]
    full_matrix['umap_embedding_y'] = umap_embed[:, 1]

    #shuffle for randomized plotting
    # umap_shuffle = full_matrix.sample(frac=1).copy()

    for i in range(len(file_list)):
        idx = np.where(full_matrix_np == i)
        axs[1].scatter(umap_embed[idx, 0], umap_embed[idx, 1], label=file_list[i], marker='.', alpha=1)
    axs[1].set_title('B', fontsize=15, fontweight='bold')

    #plt headings
    # plt.suptitle('Dimensionality Reduction')
    plt.savefig(output_dir / 'tsne-umap.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    # UMAP features
    # Area of interest
    # upper_left = np.where((umap_embed[:, 0] < 5) & (umap_embed[:, 1] > -8.5))
    # upper_right = np.where((umap_embed[:, 0] > 10) & (umap_embed[:, 1] > -7))
    subset_cells = np.where((umap_embed[:, 0] < 11) & (umap_embed[:, 1] > -15))

    full_matrix_subset_cells = full_matrix.iloc[subset_cells[0], :]
    # replace ID with tissue type
    full_matrix_subset_cells['ID'] = full_matrix_subset_cells['ID'].replace([0, 1, 2, 3, 4], ['lymph nodes', 'spleen', 'thymus', 'large intestine', 'small intestine'])
    #save as csv
    full_matrix_subset_cells.to_csv(output_dir / 'umap_subset.csv')



    # # npSM / foreground pixels
    # # for i in range(len(qs_l)):
    # #     # r = qs_l[i]['npSM'] / (1 - qs_l[i]['fracPBG'] * pixels[i])
    # #     r = qs_l[i]['npSM']
    # #     plt.hist(r, 20, alpha=0.5, label=file_list[i])
    # # plt.legend()
    # # plt.title('Number of Cells Per 100 Square Microns Among Different Tissues')
    # # plt.ylabel('Frequency')
    # # plt.xlabel('Cells per 100 Square Microns / Number of Foreground Pixels')
    # # plt.savefig(output_dir / 'nCpsm_new.png', bbox_inches='tight')
    # # plt.clf()
    # # plt.close()
    #
    # qm
    # plt.hist(qsl, label=file_list)
    # plt.legend()
    # # plt.title('Segmentation Quality Score Among Different Tissues')
    # plt.title('A')
    # plt.ylabel('Frequency')
    # plt.xlabel('Quality Score')
    # plt.savefig(output_dir / 'qs_tissues.png', bbox_inches='tight')
    # plt.clf()
    #
    # # all = [otsu_tissue_snr, zscore_tissue_snr, otsu_tissue_snr_avg, zscore_tissue_snr_avg]
    #
    all = [otsu_tissue_snr_avg, zscore_tissue_snr_avg]
    snr_name = ['A', 'B']
    tissue_avg = []
    count = 0
    boxplot_l = []

    # t = ['Channel: ' + common_channels[i] + ' Signal to Noise Ratio Among Different Tissues']
    for i in all:
        # snr_o_z = []
        #
        # for j in i:
        #     snr_o_z.append(np.concatenate(j))
        #
        # boxplot_l.append(snr_o_z)

        # make list of common channels by tissue
        cd11c = [i[0][:, 0], i[1][:, 0], i[2][:, 0],
                 i[3][:, 0], i[4][:, 0]]
        cd21 = [i[0][:, 1], i[1][:, 1], i[2][:, 1],
                i[3][:, 1], i[4][:, 1]]
        cd4 = [i[0][:, 2], i[1][:, 2], i[2][:, 2],
               i[3][:, 2], i[4][:, 2]]
        cd8 = [i[0][:, 3], i[1][:, 3], i[2][:, 3],
               i[3][:, 3], i[4][:, 3]]
        ki67 = [i[0][:, 4], i[1][:, 4], i[2][:, 4],
                i[3][:, 4], i[4][:, 4]]
        s = [cd11c, cd21, cd4, cd8, ki67]
        # s_array = np.asarray(s)
        # tissue_avg.append(np.average(s_array))

        # out = output_dir / ('snr-avg.npy')
        # np.save(out, nparray)
        if count == 0:
            fig, axs = plt.subplots(1, len(s), sharey=True, figsize=(20, 10))
        else:
            fig, axs = plt.subplots(1, len(s), figsize=(20, 10))

        for j in range(len(s)):
            # plt.subplot(1, len(s), j + 1)
            # box n whisker plots
            axs[j].boxplot(s[j], showfliers=False)
            axs[j].set_xticks([1, 2, 3, 4, 5],
                       ['lymph nodes', 'spleen', 'thymus', 'large\n intestine', 'small\n intestine'], fontsize=12, rotation=45, fontweight='bold')
            # axs[j].set_xticks(rotation=45, fontsize=8)
            axs[j].set_title(common_channels[j], fontsize=12, fontweight='bold')
            if count == 0:
                axs[j].set_yscale('log')
                # axs[0].set_ylabel('log', fontsize=12, fontweight='bold')
        # plt.ylabel('SNR per Positive Cell')
        # plt.xlabel('Tissues')
        # fig.text(0.5, 0.01, 'Tissues', ha='center', va='center', fontsize=15, fontweight='bold')
        if count == 0:
            fig.text(0.04, 0.5, 'log SnR', ha='center', va='center', rotation='vertical', fontsize=20, fontweight='bold')
        else:
            fig.text(0.04, 0.5, 'SnR', rotation='vertical', fontsize=20,
                     fontweight='bold')
        fig.suptitle(snr_name[count], fontweight='bold', fontsize=25)
        fig.savefig(output_dir / (snr_name[count] + '-boxplot.png'), dpi=800)
        fig.clf()

        plt.close(fig)

        count += 1
################################################################################
        # for i in range(len(s)):
        #     hist, bins, _ = plt.hist(s[i], bins=8)
        #     plt.clf()
        #     plt.close()
        #     logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 3)
        #     plt.hist(s[i], label=file_list, bins=logbins)
        #     plt.legend()
        #     plt.title('Channel: ' + common_channels[i] + ' Average Signal to Noise Ratio per Cell Among Different Tissues')
        #     plt.ylabel('Number of Images')
        #     plt.xlabel('Signal to Noise Ratio')
        #     plt.xscale('log')
        #     plt.savefig(output_dir / (common_channels[i] + '-avg_s2n_common.png'), bbox_inches='tight')
        #     plt.clf()
        # plt.close()

        # pca s2n
        # flat_l = []
        # for i in range(len(s)):
        #     flat_l.append(np.concatenate(s[i]))
        # flat_l = np.asarray(flat_l).T
        #
        # flat_l = stats.zscore(flat_l)
        # # PCA
        # m = PCA(n_components=2, svd_solver="full").fit(flat_l)
        # snr_pca = m.transform(flat_l)
        #
        # #manual
        # idx_c = [17, 13, 8, 30, 31]
        # for i in range(flat_l.shape[1]):
        #     if i == 0:
        #         plt.scatter(snr_pca[:idx_c[i], 0], snr_pca[:idx_c[i], 1], label=file_list[i], alpha=0.5)
        #     else:
        #         plt.scatter(snr_pca[idx_c[i-1]:idx_c[i]+idx_c[i-1], 0], snr_pca[idx_c[i-1]:idx_c[i]+idx_c[i-1], 1], label=file_list[i], alpha=0.5)
        # plt.title('SNR PCA')
        # # plt.xlabel('PC1: 76%')
        # # plt.ylabel('PC2: 23%')
        # plt.xlabel('PC1: ' + str(m.explained_variance_ratio_[0]) )
        # plt.ylabel('PC2: ' + str(m.explained_variance_ratio_[1]) )
        # plt.legend()
        # plt.savefig(output_dir / 'pca_avg_s2n_common.png', bbox_inches='tight')
        # plt.clf()
        # plt.close()




    # quality metrics pca w/ and w/o segmentation quality score
    qual_metrics_df = pd.concat(qs_l)
    qual_metrics = qual_metrics_df.to_numpy()
    #preprocess features
    features = StandardScaler().fit_transform(qual_metrics)

    tissue_length = [len(x) for x in qs_l]

    # image_qm = PCA(n_components=2, svd_solver="full").fit(qual_metrics)
    # image_qm_pca_comp = image_qm.transform(features)
    reducer = umap.UMAP()
    # umap_features = full_matrix_pop.copy()
    qm_umap_embed = reducer.fit_transform(features)

    markers = ['o', 's', 'P', 'X', 'D']
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for i in range(len(tissue_length)):
        if i == 0:
            axs[0].scatter(qm_umap_embed[:tissue_length[i], 0], qm_umap_embed[:tissue_length[i], 1], label=file_list[i], alpha=0.7, marker=markers[i], s=100)
        elif i == 1 or i == 2:
            axs[0].scatter(qm_umap_embed[tissue_length[i - 1]:tissue_length[i] + tissue_length[i - 1], 0],
                           qm_umap_embed[tissue_length[i - 1]:tissue_length[i] + tissue_length[i - 1], 1],
                           label=file_list[i], alpha=0.7, marker=markers[i], s=200)
        else:
            axs[0].scatter(qm_umap_embed[tissue_length[i-1]:tissue_length[i]+tissue_length[i-1], 0], qm_umap_embed[tissue_length[i-1]:tissue_length[i]+tissue_length[i-1], 1], label=file_list[i], alpha=0.7, marker=markers[i], s=100)


    #find loadings for umap
    data_df = pd.DataFrame(features, columns=[f'Feature {i}' for i in range(1, features.shape[1] + 1)])
    embedding_df = pd.DataFrame(qm_umap_embed, columns=[f'Component {i}' for i in range(1, qm_umap_embed.shape[1] + 1)])

    # Initialize a DataFrame to store the correlations
    correlations = pd.DataFrame(index=data_df.columns, columns=embedding_df.columns)

    # Compute the Spearman correlation between each original feature and each UMAP component
    for feature in data_df.columns:
        for component in embedding_df.columns:
            correlation, _ = spearmanr(data_df[feature], embedding_df[component])
            correlations.loc[feature, component] = correlation



    # plt.title('Image Quality Metrics PCA')
    # plt.xlabel('PC1: 76%')
    # plt.ylabel('PC2: 23%')
    # axs[0].xlabel('PC1: ' + str(image_qm.explained_variance_ratio_[0]))
    # axs[0].ylabel('PC2: ' + str(image_qm.explained_variance_ratio_[1]))
    # axs[0].legend()
    # plt.savefig(output_dir / 'image_quality_metrics_pca.png', bbox_inches='tight')
    # plt.clf()
    # plt.close()

    #w/o segmentation quality score
    # qual_metrics_drop = qual_metrics.drop(columns=['SegQS'])
    # qual_metrics_drop = np.delete(qual_metrics, 1, axis=1)
    # features_drop = StandardScaler().fit_transform(qual_metrics_drop)
    #
    # image_qm_drop = PCA(n_components=2, svd_solver="full").fit(qual_metrics_drop)
    # image_qm_pca_comp_drop = image_qm_drop.transform(features_drop)
    #
    # for i in range(len(tissue_length)):
    #     if i == 0:
    #         axs[1].scatter(image_qm_pca_comp_drop[:tissue_length[i], 0], image_qm_pca_comp[:tissue_length[i], 1], label=file_list[i], alpha=0.7, marker=markers[i], s=100)
    #     elif i == 1 or i == 2:
    #         axs[1].scatter(image_qm_pca_comp_drop[tissue_length[i - 1]:tissue_length[i] + tissue_length[i - 1], 0],
    #                        image_qm_pca_comp_drop[tissue_length[i - 1]:tissue_length[i] + tissue_length[i - 1], 1],
    #                        label=file_list[i], alpha=0.7, marker=markers[i], s=200)
    #     else:
    #         axs[1].scatter(image_qm_pca_comp_drop[tissue_length[i-1]:tissue_length[i]+tissue_length[i-1], 0], image_qm_pca_comp_drop[tissue_length[i-1]:tissue_length[i]+tissue_length[i-1], 1], label=file_list[i], alpha=0.7, marker=markers[i], s=100)



    # first_pc_drop = image_qm_drop.components_[0]
    # second_pc_drop = image_qm_drop.components_[1]
    #
    # pc1_feature_indicies_drop = np.abs(first_pc_drop).argsort()
    # pc2_feature_indicies_drop = np.abs(second_pc_drop).argsort()

    # pc1_sorted_features_drop = qual_metrics.columns[pc1_feature_indicies_drop]
    # pc2_sorted_features_drop = qual_metrics.columns[pc2_feature_indicies_drop]


    # axs[1].xlabel('PC1: ' + str(image_qm.explained_variance_ratio_[0]))
    # axs[1].ylabel('PC2: ' + str(image_qm.explained_variance_ratio_[1]))

    axs[0].set_title('A', fontsize=20, fontweight='bold')
    axs[1].set_title('B', fontsize=20, fontweight='bold')
    axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    fig.text(0.5, 0.04, 'PC1', ha='center', va='center', fontsize=20)
    fig.text(0.04, 0.5, 'PC2', ha='center', va='center', rotation='vertical', fontsize=20)
    # fig.text(1, 0.04, 'B', ha='center', va='center', fontsize=20)
    # fig.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    # fig.suptitle(snr_name[count] + '-SNR for Positive Cells')
    fig.savefig(output_dir / 'PCA-QM.png', bbox_inches="tight", dpi=500)
    fig.clf()

    plt.close(fig)

    print('END')


    #     # tsne-init
    #     # perplex = 35
    #     # tsne_header = [str(1) + "st PC", str(2) + "nd PC"]
    #     # n_iter = 1000
    #     # lr = 1
    #     #
    #     # # print(covar_matrix.shape)
    #     # # print(total_matrix.shape)
    #     # # print(mean_all_matrix.shape)
    #     # # print(shape_matrix.shape)
    #     #
    #     # full_matrix = np.concatenate((covar_matrix, total_matrix, mean_all_matrix, shape_matrix), axis=1)
    #     # ee = len(full_matrix) / 10
    #     #
    #     # # prune for infs and nans
    #     # df = pd.DataFrame(full_matrix)
    #     # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #     # df.fillna(0, inplace=True)
    #     # full_matrix = df.to_numpy()
    #
    #     # save csv of the full matrix
    #     # out = output_dir / (file_list[count] + '_full.zip')
    #     # df.to_csv(out, index=False)
    #
    #     #sample down
    #     # N = 100000
    #     # rndperm = rng.permutation(df.shape[0])
    #     # df_subset = df.loc[rndperm[:N], :].copy()
    #     #
    #     # full_matrix = df_subset.to_numpy()
    #     #
    #     # # full_matrix_l.append(full_matrix)
    #     #
    #     # # PCA
    #     # full_matrix_og = full_matrix.copy()
    #     # tries = 0
    #     # while True:
    #     #     try:
    #     #         m = PCA(n_components=2, svd_solver="full").fit(full_matrix)
    #     #         break
    #     #     except Exception as e:
    #     #         print(e)
    #     #         print("Exceptions caught: ", tries)
    #     #         if tries == 0:
    #     #             m = PCA(n_components=2, svd_solver="randomized").fit(full_matrix)
    #     #             tries += 1
    #     #         else:
    #     #             print("halving the features in tSNE for PCA fit...")
    #     #             n_samples = int(full_matrix.shape[0] / 2)
    #     #             idx = np.random.choice(
    #     #                 full_matrix_og.shape[0], n_samples, replace=False
    #     #             )
    #     #             full_matrix = full_matrix_og[idx, :]
    #     #
    #     # full_matrix = m.transform(full_matrix_og)
    #     #
    #     # # get 2D PCA
    #     # pca_l.append(full_matrix)
    #
    #     # #get kmeans cluster
    #     # nc, pca_labels = get_silhouette_score(full_matrix, 'PCA-silhouette-scores', output_dir)
    #     #
    #     # #plot 2D PCA
    #     # plt.scatter(full_matrix[:, 0], full_matrix[:, 1], c=pca_labels)
    #     # plt.legend()
    #     #
    #     #
    #     #
    #     # # tsne
    #     # full_matrix_tsne = full_matrix.copy()
    #     # tsne = TSNE(
    #     #     n_components=2,
    #     #     perplexity=perplex,
    #     #     early_exaggeration=ee,
    #     #     learning_rate=lr,
    #     #     n_iter=n_iter,
    #     #     init='pca',
    #     #     random_state=42
    #     # )
    #     #
    #     # while True:
    #     #     try:
    #     #         tsne_all = tsne.fit_transform(full_matrix)
    #     #         break
    #     #     except Exception as e:
    #     #         print(e)
    #     #         print("halving dataset in tSNE for tSNE fit...")
    #     #         n_samples = int(full_matrix.shape[0] / 2)
    #     #         idx = np.random.choice(full_matrix_tsne.shape[0], n_samples, replace=False)
    #     #         full_matrix = full_matrix_tsne[idx, :]
    #     #
    #     # # cluster tSNE
    #     # # K = range(1, 10)
    #     # # find optimal k cluster
    #     # num_cluster, tsne_labels = get_silhouette_score(tsne_all, 'tSNE-silhouette-scores', output_dir)
    #     #
    #     # # tsne_cluster = KMeans(n_clusters=3, random_state=42).fit(tsne_all)
    #     # # tsne_labels = tsne_cluster.labels_
    #     #
    #     # # df = pd.DataFrame(tsne_all)
    #     # # f = output_dir / (file_list[count] + '-tsne.csv')
    #     # # df.to_csv(f)
    #     #
    #     # # add random sampling - don't need all the points
    #     #
    #     # f2 = output_dir / (file_list[count] + '-tsne_plot.png')
    #     # plt.scatter(tsne_all[:, 0], tsne_all[:, 1], c=tsne_labels, marker='.')
    #     # plt.savefig(f2, format='png')
    #     # plt.clf()
    #
    #     count += 1
    #
    #
    # #do pca on subsample of all tissues
    # # full_matrix_l = np.asarray(full_matrix_l)
    # #
    # # N = 20000
    # # labels_t = np.arange(len(file_list))
    # # labels_t = np.repeat(labels_t, N)
    # # mixed_matrix = np.zeros(())
    # #
    # # for i in len(range(file_list)):
    # #     full_matrix_t = full_matrix_l[i]
    # #     rndperm = rng.permutation(full_matrix_t.shape[0])
    # #     subset = full_matrix_t[rndperm[:N], :].copy()
    #
    # # make list of common channels by tissue
    # cd11c = [common_tissue_snr[0][:, 0], common_tissue_snr[1][:, 0], common_tissue_snr[2][:, 0],
    #          common_tissue_snr[3][:, 0], common_tissue_snr[4][:, 0]]
    # cd21 = [common_tissue_snr[0][:, 1], common_tissue_snr[1][:, 1], common_tissue_snr[2][:, 1],
    #         common_tissue_snr[3][:, 1], common_tissue_snr[4][:, 1]]
    # cd4 = [common_tissue_snr[0][:, 2], common_tissue_snr[1][:, 2], common_tissue_snr[2][:, 2],
    #        common_tissue_snr[3][:, 2], common_tissue_snr[4][:, 2]]
    # cd8 = [common_tissue_snr[0][:, 3], common_tissue_snr[1][:, 3], common_tissue_snr[2][:, 3],
    #        common_tissue_snr[3][:, 3], common_tissue_snr[4][:, 3]]
    # ki67 = [common_tissue_snr[0][:, 4], common_tissue_snr[1][:, 4], common_tissue_snr[2][:, 4],
    #         common_tissue_snr[3][:, 4], common_tissue_snr[4][:, 4]]
    # s = [cd11c, cd21, cd4, cd8, ki67]
    # nparray = np.asarray(s)
    # out = output_dir / ('snr-avg.npy')
    # np.save(out, nparray)
    #
    # for i in range(len(s)):
    #     plt.hist(s[i], label=file_list)
    #     plt.legend()
    #     plt.title('Channel: ' + common_channels[i] + ' Signal to Noise Ratio Among Different Tissues')
    #     plt.ylabel('Frequency')
    #     plt.xlabel('Average Signal to Noise Ratio')
    #     plt.savefig(output_dir / (common_channels[i] + '-s2n_common.png'), bbox_inches='tight')
    #     plt.clf()
    #
    # # rng = np.random.default_rng(42)
    # N = 20000
    # #init
    # fig0, ax0 = plt.subplots()
    # # fig1, ax1 = plt.subplots()
    # #pca 2d plot - sample 20000 points to plot and also full rez version
    # for i in range(len(pca_l)):
    #     #resample
    #     rndperm = rng.permutation(pca_l[i].shape[0])
    #     sample = pca_l[i][rndperm[:N], :]
    #     ax0.scatter(sample[:, 0], sample[:, 1], label=file_list[i])
    #     # ax1.scatter(pca_l[i][:, 0], pca_l[i][:, 1], label=file_list[i])
    #
    # ax0.set_ylabel('PC 2')
    # ax0.set_xlabel('PC 1')
    # # ax1.set_ylabel('PC 2')
    # # ax1.set_xlabel('PC 1')
    # ax0.legend()
    # # ax1.legend()
    #
    # fig0.suptitle('PC1 vs. PC2 - All Tissues', fontsize=16)
    # fig0.savefig(output_dir / 'PCA_allTissues.png', bbox_inches='tight')
    # exit()
    #
    # # fig1.suptitle('', fontsize=16)
    # # ax1.tight_layout()
    # # fig1.savefig(output_dir / ())
    #
    #
    # plt.close(fig0)
    # plt.close(fig1)
    #
    #
    #
    # tissue_df = []
    # channels = []
    # # filter channels together
    # for i in range(len(tissue_snr)):
    #     all_channels = pd.concat(tissue_snr[i], axis=0, ignore_index=True)
    #     # all_channels.fillna(0, inplace=True)
    #     tissue_df.append(all_channels)
    #     channels.append(all_channels.columns.tolist())
    #
    # # make dfs
    # ln_df = pd.DataFrame(tissue_df[0], columns=channels[0])
    # spleen_df = pd.DataFrame(tissue_df[1], columns=channels[1])
    # thymus_df = pd.DataFrame(tissue_df[2], columns=channels[2])
    # largeint_df = pd.DataFrame(tissue_df[3], columns=channels[3])
    # smallint_df = pd.DataFrame(tissue_df[4], columns=channels[4])
    #
    # all_channels_t = np.concatenate(channels)
    # unique_channels = np.unique(all_channels_t)
    #
    # # tot = 5
    # # cols = len(unique_channels)
    # # rows = tot // cols
    # # rows += tot % cols
    # for i in range(len(unique_channels)):
    #     # fig = plt.figure(i)
    #     ln_f = ln_df.filter(regex=unique_channels[i])
    #     spleen_f = spleen_df.filter(regex=unique_channels[i])
    #     thymus_f = thymus_df.filter(regex=unique_channels[i])
    #     largeint_f = largeint_df.filter(regex=unique_channels[i])
    #     smallint_f = smallint_df.filter(regex=unique_channels[i])
    #
    #     l = [ln_f, spleen_f, thymus_f, largeint_f, smallint_f]
    #
    #     tot = 0
    #     skipped = []
    #     for j in range(len(l)):
    #         if unique_channels[i] in l[j].columns:
    #             tot += 1
    #         else:
    #             skipped.append(j)
    #
    #     idx = 0
    #     # box and whisker
    #     fig, axs = plt.subplots(1, tot, sharey=True)
    #     # ax = fig.add_subplot(1, tot, idx, sharey=True)
    #     for j in range(len(l)):
    #         y = 0
    #         if j not in skipped:
    #             # filter out NaNs
    #             filterd = l[j].to_numpy()
    #             filterd = filterd[~np.isnan(filterd)]
    #
    #             if tot > 1:
    #                 axs[idx].boxplot(filterd)
    #                 axs[idx].set_title(file_list[j])
    #                 if y == 0:
    #                     axs[idx].set_ylabel('Average Signal to Noise Ratio')
    #                 idx += 1
    #                 y += 1
    #             else:
    #                 axs.boxplot(filterd)
    #                 axs.set_title(file_list[j])
    #                 axs.set_ylabel('Average Signal to Noise Ratio')
    #
    #     fig.suptitle('Channel: ' + unique_channels[i], fontsize=16)
    #     plt.tight_layout()
    #     fig.savefig(output_dir / (unique_channels[i] + '.png'), bbox_inches='tight')
    #     fig.clf()


def get_silhouette_score(d, s, o):
    n = np.arange(2, 11)
    silhouette_avg = []
    kmeans_l = []
    for num_clusters in n:
        # initialise kmeans
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(d)
        cluster_labels = kmeans.labels_

        kmeans_l.append(cluster_labels)

        # silhouette score
        silhouette_avg.append(silhouette_score(d, cluster_labels))

    plt.plot(n, silhouette_avg, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette analysis to find optimal clusters for ')
    plt.tight_layout()
    plt.savefig(o / (s + '.png'), bbox_inches='tight')
    plt.clf()

    idx = np.argmax(silhouette_avg)

    return idx + 2, kmeans_l[idx]


def find_common_channels(img_l):
    x = set(img_l[0].columns)
    for i in range(len(img_l) - 1):
        x = x & set(img_l[i + 1].columns)

    # filter out cell id
    x.remove('ID')

    return list(x)


def filter_common_channels(ccn, paths):
    df = [pd.read_csv(x) for x in paths]
    # filter to get only common channels
    l = [x[ccn] for x in df]
    f = pd.concat(l, ignore_index=True)

    return f.to_numpy()


def qm_process(paths, common_channels):
    dict_l = []
    qsl = []
    alldf = []
    nCpsm_l = []

    # make df for quality metrics
    df = pd.DataFrame(columns=['1/AvgCVBackground', 'FracPixBG', 'FracImgCells', 'NumCells', 'NumChannels', 'AvgCellBG',
                               'AvgNucCells', 'AvgTotInt'])

    for i in paths:
        # qm = json.load(i.read_bytes())
        with open(i) as f:
            qm = json.load(f)

        s2n_otsu = qm['Image Quality Metrics not requiring image segmentation']['Signal To Noise Otsu']
        s2n_z = qm['Image Quality Metrics not requiring image segmentation']['Signal To Noise Z-Score']
        qs = qm['Segmentation Evaluation Metrics']['QualityScore']
        nCpsm = qm['Segmentation Evaluation Metrics']['Matched Cell']['NumberOfCellsPer100SquareMicrons']

        # for quality metrics
        nChannels = qm['Image Information']['Number of Channels']
        invAvgcbg = qm['Image Quality Metrics requiring background segmentation']['1/AvgCVBackground']
        fracPBG = qm['Image Quality Metrics requiring background segmentation'][
            'Fraction of Pixels in Image Background']
        fracImgCells = qm['Image Quality Metrics that require cell segmentation']['Fraction of Image Occupied by Cells']
        nCells = qm['Image Quality Metrics that require cell segmentation']['Number of Cells']

        # iterate through keys
        cBG = 0
        nucCell = 0
        avgCellR = qm['Image Quality Metrics that require cell segmentation']['Channel Statistics'][
            'Average per Cell Ratios']
        for key in avgCellR.values():
            cBG += key['Cell / Background']
            nucCell += key['Nuclear / Cell']
        # avg
        avgcBG = cBG / nChannels
        avgnucCell = nucCell / nChannels

        # total intensity
        totInt = 0
        totIntC = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']
        for val in totIntC.values():
            totInt += val

        # avg totint
        totInt = totInt / nChannels

        # append to df
        to_append = [invAvgcbg, fracPBG, fracImgCells, nCells, nChannels, avgcBG, avgnucCell, totInt]
        series_append = pd.Series(to_append, index=df.columns)
        df = df.append(series_append, ignore_index=True)

        # common channel names
        common_otsu = [s2n_otsu['CD11c'], s2n_otsu['CD21'], s2n_otsu['CD4'], s2n_otsu['CD8'], s2n_otsu['Ki67']]
        common_z = [s2n_z['CD11c'], s2n_z['CD21'], s2n_z['CD4'], s2n_z['CD8'], s2n_z['Ki67']]
        a = np.asarray(common_otsu)
        b = np.asarray(common_z)

        c = (a + b) / 2
        dict_l.append(c)
        # dict_l = np.asarray(dict_l)

        qsl.append(qs)

        a = np.asarray(list(s2n_otsu.values()))
        b = np.asarray(list(s2n_z.values()))
        c = (a + b) / 2
        c = c.reshape(1, len(c))

        df_2 = pd.DataFrame(c, columns=s2n_otsu.keys())

        alldf.append(df_2)
        nCpsm_l.append(nCpsm)

    return np.asarray(dict_l), qsl, alldf, nCpsm_l, df


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


def box_n_whisker():
    pass


if __name__ == "__main__":
    main()
