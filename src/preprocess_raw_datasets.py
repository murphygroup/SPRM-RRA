from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import umap
from skimage import measure
import pickle
import os
# from aicsimageio import AICSImage
# from PIL import Image
import tifffile


def main():
    #remote
    root = Path('/hive/users/tedz/workspace/REPROCESS_CODEX/')
    stanford = Path('/hive/hubmap/data/consortium/Stanford TMC/')
    bfconvert = Path('/hive/users/tedz/workspace/BFCONVERTED/reprocess-v2.4.2/')

    #local
    # root = Path('/Users/tedzhang/Desktop/CMU/murphylab/SPRM/MANUSCRIPT/preprocess_sprm_outputs/')
    # test = Path('/Users/tedzhang/Desktop/CMU/murphylab/SPRM/TEST/CODEX_HANG_20230216/mask/reg001_mask.ome.tiff')

    #
    lymphnodes = root / 'ln'
    spleen = root / 'spleen'
    thymus = root / 'thymus'
    largeint = root / 'li'
    smallint = root / 'si'

    tissue_list = [lymphnodes, spleen, thymus, largeint, smallint]
    # tissue_list = [smallint]
    file_list = ['lymph_nodes', 'spleen', 'thymus', 'large_intestine', 'small_intestine']
    # file_list = ['small_intestine']
    # pixels = [12664 * 7491, 9975 * 9489, 9973 * 9492, 10003 * 9529, 9997 * 9521]

    tissue_pixels = {}
    count = 0

    output_dir = Path.home() / 'workspace' / 'FIGURES' / 'input'
    # output_dir = Path('/Users/tedzhang/Desktop/CMU/murphylab/SPRM/MANUSCRIPT/input_test')

    # list inits
    common_tissue_snr = []
    # qsl = []
    tissue_snr = []
    common_channels = ['CD11c', 'CD21', 'CD4', 'CD8', 'Ki67']
    # nCpsm_l = []

    # pca list
    pca_l = []
    # tsne_l = []
    ccn_l = []
    tissue_feats = []

    # random init
    rng = np.random.default_rng(42)
    N = 20000

    # names = ['covar', 'total', 'mean-all', 'shape', 'total_cells']
    names = ['covar_cells', 'mean_cells']
    for i in tissue_list:
        # search for feature files
        mean_pathlist = list(i.glob(r'**/*_channel_mean.csv'))
        covar_pathlist = list(i.glob(r'**/*_channel_covar.csv'))
        # total_pathlist = list(i.glob(r'**/*_channel_total.csv'))
        # shape_pathlist = list(i.glob(r'**/*-cell_shape.csv'))
        # meanall_pathlist = list(i.glob(r'**/*_channel_meanAll.csv'))
        # qm_pathlist = list(i.glob(r'**/*.json'))
        # cell_polygons_pathlist = list(i.glob(r'**/*-cell_polygons_spatial.csv'))
        # texture not included since was turned off for this codex dataset release

        #get the dataset id
        # dataset_ids = os.listdir(i)
        # pixels = []

        # print(dataset_ids)

        # for id in dataset_ids:

        #     # if count == 3 or count == 4:
        #     if count == 0 or count == 1:
        #         # mask_path = stanford / id / 'pipeline_output' / 'mask' / 'reg001_mask.ome.tiff'
        #         mask_path_root = stanford / id 
        #         # print(mask_path)
        #         # mask_path = str(mask_path)
        #         # try:
        #         #     # mask_path = list(mask_path_root.glob(r'**/reg001_mask.ome.tiff'))[0]
        #         #     mask = tifffile.imread(mask_path)
        #         # except:
        #         #     mask_path = stanford / id / 'stitched' / 'mask' / 'reg1_stitched_mask.ome.tiff'
        #         #     mask_path = str(mask_path)
        #         #     # mask_path = list(mask_path_root.glob(r'**/reg1_stitched_mask.*'))[0]
        #         #     mask = tifffile.imread(mask_path)

        #         mask_path = list(mask_path_root.glob(r'**/*_mask.*'))[0]   
                
        #     else:
        #         # mask_paths = bfconvert / id / 'pipeline_output' / 'mask' / 'reg001_mask.ome.tiff_new.ome.tiff'
        #         mask_path_root = bfconvert / id / 'pipeline_output' 
        #         mask_path = list(mask_path_root.glob(r'**/reg001_mask.*'))[0]        

        #     mask = tifffile.imread(mask_path)
        #     resolution = mask.shape[-1] * mask.shape[-2]
        #     pixels.append(resolution)

        # # print(len(pixels))
        # # print(len(dataset_ids))
        # assert len(pixels) == len(dataset_ids)
        # #make into dict
        # tissue_pixels = dict(zip(dataset_ids, pixels))
        # #save pickle
        # f = open(output_dir / (file_list[count] + "-pixels.pkl"),"wb")
        # # write the python object (dict) to pickle file
        # pickle.dump(tissue_pixels,f)
        # # close file
        # f.close()

        # get qm data
        # snr_common, qs_df, tot_l = qm_process(qm_pathlist, common_channels, tissue_pixels)
        # common_tissue_snr.append(snr_common)

        # save snr_common & tot_l
        # out = output_dir / (file_list[count] + '-snr_common.npy')
        # np.save(out, snr_common)
        # count += 1
        # continue
    
        #
        # out = output_dir / (file_list[count] + '-total_intensity_common.npy')
        # np.save(out, tot_l)

        # qs_df.to_pickle(output_dir / (file_list[count] + '-qs.pkl'))
        # count += 1
        # continue

        # seg_list = [mean_pathlist, covar_pathlist, total_pathlist]

        mean_paths = feature_seg(mean_pathlist)
        covar_paths = feature_seg(covar_pathlist)
        # total_paths = feature_seg(total_pathlist)

        # seg_list = [total_paths, covar_paths]
        seg_list = [mean_paths, covar_paths]


        c = []
        for j in range(len(seg_list)):
            a = []
            for k in range(len(seg_list[j])):
                l = [pd.read_csv(t) for t in seg_list[j][k]]

                # print(len(l))
                # print(seg_list[j][k])

                # find common channels
                if k == 0:
                    ccn_t = find_common_channels(l)
                    # if j == 0:
                    #     ccn_mean = ccn_t

                # filter to get only common channels
                l = [x[ccn_t] for x in l]

                #add pixel information
                # if j == 0 and k == 0:
                #     for p in range(len(l)):
                #         l[p]['pixels'] = pixels[p]

                f = pd.concat(l, ignore_index=True)
                a.append(f)
            # a = np.asarray(a)
            c.append(a)

        # common channel names within tissues
        # ccn_l.append(ccn_mean)
        # # save
        # with open(output_dir / (file_list[count] + '-channel_mean_names.pkl'), 'wb') as f:
        #     pickle.dump(ccn_mean, f)

        mean_matrix = c[0]
        # covar_matrix = c[1]
        # total_matrix = c[0]
        covar_matrix = c[1]

        # mean_all_matrix = filter_common_channels(ccn_mean, meanall_pathlist)
        # shape_matrix = pd.concat([pd.read_csv(t) for t in shape_pathlist], ignore_index=True)

        #get snr per image
        # _ = snr_cells(pixels, snr_common, tot_l)

        # just one column of a list coordinates
        # cell_polygons = pd.concat([pd.read_csv(t) for t in cell_polygons_pathlist], ignore_index=True).to_numpy()

        # _ = outlinePCA(cell_polygons)

        # reformat covar and total
        # covar_matrix = covar_matrix.reshape(
        #     (covar_matrix.shape[0], covar_matrix.shape[1], covar_matrix.shape[2] * covar_matrix.shape[3]))
        # covar_matrix = pd.concat(covar_matrix, axis=1)

        #get cells by themselves
        # total_matrix_cells = total_matrix[0]
        covar_matrix_cells = covar_matrix[1]
        mean_matrix_cells = mean_matrix[0]

        # total_matrix = total_matrix.reshape(
        #     (total_matrix.shape[0], total_matrix.shape[1], total_matrix[2] * total_matrix[3]))
        # total_matrix = pd.concat(total_matrix[1:], axis=1)



        # recreate mean all
        # mean_all_matrix = np.concatenate(mean_matrix, axis=1)

        # save down sample of all features for local processing
        # all_feats = [covar_matrix, total_matrix, mean_all_matrix, shape_matrix, total_matrix_cells]
        all_feats = [covar_matrix_cells, mean_matrix_cells]

        all_feat_ds_l = []
        # N = 20000
        rndperm = rng.permutation(covar_matrix_cells.shape[0])

        # print(shape_matrix.shape)
        # print(total_matrix.shape)

        for j in range(len(all_feats)):
            # print(all_feats[j].shape)
            # print(all_feats[j])
            subset_matrix = all_feats[j].loc[rndperm[:N], :].copy()
            out = output_dir / (file_list[count] + '-' + names[j])
            subset_matrix.to_csv(out)
            all_feat_ds_l.append(subset_matrix)
        
        print('finish ' + file_list[count])
        count += 1
        continue

        # print(covar_matrix.shape)
        # print(total_matrix.shape)
        # print(mean_all_matrix.shape)
        # print(shape_matrix.shape)
        #
        all_feats = pd.concat(all_feat_ds_l)
        # full_matrix = np.concatenate((covar_matrix, total_matrix, mean_all_matrix, shape_matrix), axis=1)

        # prune for infs and nans
        # df = pd.DataFrame(full_matrix)
        all_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        all_feats.fillna(0, inplace=True)
        # full_matrix = df.to_numpy()

        #add id for which tissue the full feature set belongs to
        tissue_idx = np.zeros(N)
        tissue_idx[:] = count
        tissue_idx = tissue_idx.tolist()
        all_feats['tissue_id'] = tissue_idx


        #per tissue full features
        tissue_feats.append(all_feats)

        # save csv of the full matrix
        # out = output_dir / (file_list[count] + '_full.zip')
        # df.to_csv(out, index=False)

        # sample down
        # N = 20000
        # rndperm = rng.permutation(df.shape[0])
        # df_subset = df.loc[rndperm[:N], :].copy()
        #
        # full_matrix = df_subset.to_numpy()
        #
        # full_matrix_l.append(full_matrix)

        # # PCA
        # full_matrix_og = full_matrix.copy()
        # tries = 0
        # while True:
        #     try:
        #         m = PCA(n_components=2, svd_solver="full").fit(full_matrix)
        #         break
        #     except Exception as e:
        #         print(e)
        #         print("Exceptions caught: ", tries)
        #         if tries == 0:
        #             m = PCA(n_components=2, svd_solver="randomized").fit(full_matrix)
        #             tries += 1
        #         else:
        #             print("halving the features in tSNE for PCA fit...")
        #             n_samples = int(full_matrix.shape[0] / 2)
        #             idx = np.random.choice(
        #                 full_matrix_og.shape[0], n_samples, replace=False
        #             )
        #             full_matrix = full_matrix_og[idx, :]
        #
        # full_matrix = m.transform(full_matrix_og)
        #
        # # get 2D PCA
        # pca_l.append(full_matrix)

        # #get kmeans cluster
        # nc, pca_labels = get_silhouette_score(full_matrix, 'PCA-silhouette-scores', output_dir)
        #
        # #plot 2D PCA
        # plt.scatter(full_matrix[:, 0], full_matrix[:, 1], c=pca_labels)
        # plt.legend()
        #
        #
        #
        # # tsne
        # full_matrix_tsne = full_matrix.copy()
        # tsne = TSNE(
        #     n_components=2,
        #     perplexity=perplex,
        #     early_exaggeration=ee,
        #     learning_rate=lr,
        #     n_iter=n_iter,
        #     init='pca',
        #     random_state=42
        # )
        #
        # while True:
        #     try:
        #         tsne_all = tsne.fit_transform(full_matrix)
        #         break
        #     except Exception as e:
        #         print(e)
        #         print("halving dataset in tSNE for tSNE fit...")
        #         n_samples = int(full_matrix.shape[0] / 2)
        #         idx = np.random.choice(full_matrix_tsne.shape[0], n_samples, replace=False)
        #         full_matrix = full_matrix_tsne[idx, :]
        #
        # # cluster tSNE
        # # K = range(1, 10)
        # # find optimal k cluster
        # num_cluster, tsne_labels = get_silhouette_score(tsne_all, 'tSNE-silhouette-scores', output_dir)
        #
        # # tsne_cluster = KMeans(n_clusters=3, random_state=42).fit(tsne_all)
        # # tsne_labels = tsne_cluster.labels_
        #
        # # df = pd.DataFrame(tsne_all)
        # # f = output_dir / (file_list[count] + '-tsne.csv')
        # # df.to_csv(f)
        #
        # # add random sampling - don't need all the points
        #
        # f2 = output_dir / (file_list[count] + '-tsne_plot.png')
        # plt.scatter(tsne_all[:, 0], tsne_all[:, 1], c=tsne_labels, marker='.')
        # plt.savefig(f2, format='png')
        # plt.clf()

        count += 1

    print('finish looping tissues')
    # exit
    exit()

    #concat all tissue full features
    tissue_feats = pd.concat(tissue_feats, axis=0, ignore_index=True)

    #set NaN to 0
    tissue_feats.fillna(0, inplace=True)

    # tsne-init
    perplex = 35
    tsne_header = [str(1) + "st PC", str(2) + "nd PC"]
    n_iter = 1000
    lr = 1
    ee = len(tissue_feats) / 10

    # do pca on subsample of all tissues
    # full_matrix_l = np.asarray(full_matrix_l)
    #
    # N = 20000
    # labels_t = np.arange(len(file_list))
    # labels_t = np.repeat(labels_t, N)
    # mixed_matrix = np.zeros(())
    #
    # for i in len(range(file_list)):
    #     full_matrix_t = full_matrix_l[i]
    #     rndperm = rng.permutation(full_matrix_t.shape[0])
    #     subset = full_matrix_t[rndperm[:N], :].copy()

    # make list of common channels by tissue
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
    # exit()

    # for i in range(len(s)):
    #     plt.hist(s[i], label=file_list)
    #     plt.legend()
    #     plt.title('Channel: ' + common_channels[i] + ' Signal to Noise Ratio Among Different Tissues')
    #     plt.ylabel('Frequency')
    #     plt.xlabel('Average Signal to Noise Ratio')
    #     plt.savefig(output_dir / (common_channels[i] + '-s2n_common.png'), bbox_inches='tight')
    #     plt.clf()

    #zscore features


    # rng = np.random.default_rng(42)
    N = 20000
    # init
    fig0, ax0 = plt.subplots()
    # fig1, ax1 = plt.subplots()
    # pca 2d plot - sample 20000 points to plot and also full rez version
    for i in range(len(pca_l)):
        # resample
        rndperm = rng.permutation(pca_l[i].shape[0])
        sample = pca_l[i][rndperm[:N], :]
        ax0.scatter(sample[:, 0], sample[:, 1], label=file_list[i])
        # ax1.scatter(pca_l[i][:, 0], pca_l[i][:, 1], label=file_list[i])

    ax0.set_ylabel('PC 2')
    ax0.set_xlabel('PC 1')
    # ax1.set_ylabel('PC 2')
    # ax1.set_xlabel('PC 1')
    ax0.legend()
    # ax1.legend()

    fig0.suptitle('PC1 vs. PC2 - All Tissues', fontsize=16)
    fig0.savefig(output_dir / 'PCA_allTissues.png', bbox_inches='tight')

    # fig1.suptitle('', fontsize=16)
    # ax1.tight_layout()
    # fig1.savefig(output_dir / ())

    plt.close(fig0)
    plt.close(fig1)

    # qm
    plt.hist(qsl, label=file_list)
    plt.legend()
    # plt.title('Segmentation Quality Score Among Different Tissues')
    plt.title('A')
    plt.ylabel('Frequency')
    plt.xlabel('Quality Score')
    plt.savefig(output_dir / 'qs_tissues.png', bbox_inches='tight')
    plt.clf()

    # nCpsm
    # plt.hist(nCpsm_l, label=file_list)
    # plt.legend()
    # plt.title('Number of Cells Per 100 Square Microns Among Different Tissues')
    # plt.ylabel('Frequency')
    # plt.xlabel('Cells per 100 microns squared')
    # plt.savefig(output_dir / 'nCpsm.png', bbox_inches='tight')


def read_tiff_voxel_size(file_path):

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.

    with tifffile.TiffFile(file_path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get('spacing', 1.)
        else:
            # default voxel size
            z = 1.

        tags = tiff.pages[0].tags

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

    return f


def qm_process(paths, common_channels, tissue_dict):
    dict_l = []
    # qsl = []
    # alldf = []
    # nCpsm_l = []
    tot_l = []

    # make df for quality metrics
    df = pd.DataFrame(columns=['1/AvgCVBackground', 'FracImgCells', 'AvgCellBG', 'npSM', 'SegQS', 'fracPBG'])

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
        fracPBG = qm['Image Quality Metrics requiring background segmentation']['Fraction of Pixels in Image Background']
        fracImgCells = qm['Image Quality Metrics that require cell segmentation']['Fraction of Image Occupied by Cells']
        # nCells = qm['Image Quality Metrics that require cell segmentation']['Number of Cells']

        # iterate through keys
        cBG = 0
        # nucCell = 0
        avgCellR = qm['Image Quality Metrics that require cell segmentation']['Channel Statistics'][
            'Average per Cell Ratios']
        for key in avgCellR.values():
            cBG += key['Cell / Background']
            # nucCell += key['Nuclear / Cell']
        # avg
        avgcBG = cBG / nChannels
        # avgnucCell = nucCell / nChannels

        # total intensity
        # totInt = 0
        # totIntC = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']
        # for val in totIntC.values():
        #     totInt += val
        #
        # #avg totint
        # totInt = totInt / nChannels

        # append to df
        to_append = [invAvgcbg, fracImgCells, avgcBG, nCpsm, qs, fracPBG]
        series_append = pd.Series(to_append, index=df.columns)
        df = df.append(series_append, ignore_index=True)

        # common channel names
        common_otsu = [s2n_otsu['CD11c'], s2n_otsu['CD21'], s2n_otsu['CD4'], s2n_otsu['CD8'], s2n_otsu['Ki67']]
        common_z = [s2n_z['CD11c'], s2n_z['CD21'], s2n_z['CD4'], s2n_z['CD8'], s2n_z['Ki67']]
        a = np.asarray(common_otsu)
        b = np.asarray(common_z)
        #
        c = np.concatenate((a, b), axis=0)
        dict_l.append(c)
        # dict_l = np.asarray(dict_l)

        # qsl.append(qs)

        # total intensity of common channels
        totInt1 = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']['CD11c']
        totInt2 = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']['CD21']
        totInt3 = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']['CD4']
        totInt4 = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']['CD8']
        totInt5 = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']['Ki67']

        all_tot = [totInt1, totInt2, totInt3, totInt4, totInt5]
        tot_l.append(all_tot)

        # a = np.asarray(list(s2n_otsu.values()))
        # b = np.asarray(list(s2n_z.values()))
        # c = (a + b) / 2
        # c = c.reshape(1, len(c))
        #
        # df_2 = pd.DataFrame(c, columns=s2n_otsu.keys())
        #
        # alldf.append(df_2)
        # nCpsm_l.append(nCpsm)

    dict_l = np.asarray(dict_l)
    tot_l = np.asarray(tot_l)

    return dict_l, df, tot_l

def snr_cells(pixels_list, snr_common, total_list, total_percell, count):
    pass

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

    tissue_df = []
    channels = []
    # filter channels together
    for i in range(len(tissue_snr)):
        all_channels = pd.concat(tissue_snr[i], axis=0, ignore_index=True)
        # all_channels.fillna(0, inplace=True)
        tissue_df.append(all_channels)
        channels.append(all_channels.columns.tolist())

    # make dfs
    ln_df = pd.DataFrame(tissue_df[0], columns=channels[0])
    spleen_df = pd.DataFrame(tissue_df[1], columns=channels[1])
    thymus_df = pd.DataFrame(tissue_df[2], columns=channels[2])
    largeint_df = pd.DataFrame(tissue_df[3], columns=channels[3])
    smallint_df = pd.DataFrame(tissue_df[4], columns=channels[4])

    all_channels_t = np.concatenate(channels)
    unique_channels = np.unique(all_channels_t)

    # tot = 5
    # cols = len(unique_channels)
    # rows = tot // cols
    # rows += tot % cols
    for i in range(len(unique_channels)):
        # fig = plt.figure(i)
        ln_f = ln_df.filter(regex=unique_channels[i])
        spleen_f = spleen_df.filter(regex=unique_channels[i])
        thymus_f = thymus_df.filter(regex=unique_channels[i])
        largeint_f = largeint_df.filter(regex=unique_channels[i])
        smallint_f = smallint_df.filter(regex=unique_channels[i])

        l = [ln_f, spleen_f, thymus_f, largeint_f, smallint_f]

        tot = 0
        skipped = []
        for j in range(len(l)):
            if unique_channels[i] in l[j].columns:
                tot += 1
            else:
                skipped.append(j)

        idx = 0
        # box and whisker
        fig, axs = plt.subplots(1, tot, sharey=True)
        # ax = fig.add_subplot(1, tot, idx, sharey=True)
        for j in range(len(l)):
            y = 0
            if j not in skipped:
                # filter out NaNs
                filterd = l[j].to_numpy()
                filterd = filterd[~np.isnan(filterd)]

                if tot > 1:
                    axs[idx].boxplot(filterd)
                    axs[idx].set_title(file_list[j])
                    if y == 0:
                        axs[idx].set_ylabel('Average Signal to Noise Ratio')
                    idx += 1
                    y += 1
                else:
                    axs.boxplot(filterd)
                    axs.set_title(file_list[j])
                    axs.set_ylabel('Average Signal to Noise Ratio')

        fig.suptitle('Channel: ' + unique_channels[i], fontsize=16)
        plt.tight_layout()
        fig.savefig(output_dir / (unique_channels[i] + '.png'), bbox_inches='tight')
        fig.clf()



if __name__ == "__main__":
    main()
