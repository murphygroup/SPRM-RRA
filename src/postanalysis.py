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
import scipy.io
# from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib import collections as mc
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable


############




def main():
    #remote
    root = Path('/hive/users/tedz/workspace/REPROCESS_CODEX/')
    stanford = Path('/hive/hubmap/data/consortium/Stanford TMC/')
    bfconvert = Path('/hive/users/tedz/workspace/BFCONVERTED/reprocess-v2.4.2/')

    # local
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

    #adj list tissue list
    df_tissue_cell_conn = []
    adj_matrix_list = []
    cell_center_list = []
    dataset_list = []


    # random init
    rng = np.random.default_rng(42)
    N = 20000

    names = ['covar', 'total', 'mean-all', 'shape', 'total_cells']
    # names = ['covar_cells', 'mean_cells']
    for i in tissue_list:
        # search for feature files
        mean_pathlist = list(i.glob(r'**/*_channel_mean.csv'))
        covar_pathlist = list(i.glob(r'**/*_channel_covar.csv'))
        total_pathlist = list(i.glob(r'**/*_channel_total.csv'))
        shape_pathlist = list(i.glob(r'**/*-cell_shape.csv'))
        meanall_pathlist = list(i.glob(r'**/*_channel_meanAll.csv'))
        qm_pathlist = list(i.glob(r'**/*.json'))
        cell_polygons_pathlist = list(i.glob(r'**/*-cell_polygons_spatial.csv'))
        # texture not included since was turned off for this codex dataset release
        adj_matrix_pathlist = list(i.glob((r'**/*_AdjacencyMatrix.mtx')))
        cell_centers_pathlist = list(i.glob((r'**/*-cell_centers.csv')))


        #get the dataset id
        dataset_ids = os.listdir(i)
        # dataset_list.append(dataset_ids)
        # pixels = []

        # print(dataset_ids)

        # for id in dataset_ids:

        #     if count == 3 or count == 4:
        #     # if count == 0 or count == 1:
        #         # mask_path = stanford / id / 'pipeline_output' / 'mask' / 'reg001_mask.ome.tiff'
        #         mask_path_root = stanford / id 
        #         # print(mask_path)
        #         # mask_path = str(mask_path)
        #         try:
        #             mask_path = list(mask_path_root.glob(r'**/reg001_*'))[0]
        #             # mask = tifffile.imread(mask_path)
        #         except:
        #             # mask_path = stanford / id / 'stitched' / 'mask' / 'reg1_stitched_mask.ome.tiff'
        #         #     mask_path = str(mask_path)
        #             mask_path = list(mask_path_root.glob(r'**/reg1_stitched_mask.*'))
                    
        #             if mask_path:
        #                 mask_path = mask_path[0]
        #             else:
        #                 new_root = Path('/hive/hubmap/data/public/')
        #                 new_root = new_root / id
        #                 mask_path = list(new_root.glob(r'**/reg001_*'))[0]
        #         #     mask = tifffile.imread(mask_path)

        #         # mask_path = list(mask_path_root.glob(r'**/*_mask.*'))[0]   
                
        #     else:
        #         # mask_paths = bfconvert / id / 'pipeline_output' / 'mask' / 'reg001_mask.ome.tiff_new.ome.tiff'
        #         mask_path_root = bfconvert / id / 'pipeline_output' 
        #         mask_path = list(mask_path_root.glob(r'**/reg001_mask.*'))[0]        

        #     mask = tifffile.imread(mask_path)
        #     resolution = mask.shape[-1] * mask.shape[-2]
        #     pixels.append(resolution)

        # print(len(pixels))
        # print(len(dataset_ids))
        # assert len(pixels) == len(dataset_ids)
        # #make into dict
        # tissue_pixels = dict(zip(dataset_ids, pixels))
        # #save pickle
        # f = open(output_dir / (file_list[count] + "-pixels.pkl"),"wb")
        # # write the python object (dict) to pickle file
        # pickle.dump(tissue_pixels,f)
        # # close file
        # f.close()

        # read mtx matrix 
        # List to store the number of connections per row
        cell_count = []
        row_conn_append = []

        for file in cell_centers_pathlist:
            cell_center_list.append(pd.read_csv(file))
        # df = pd.DataFrame()

        # Read each .mtx file and count the non-zero entries per row
        for file in adj_matrix_pathlist:
            # Read the .mtx file
            matrix = scipy.io.mmread(file)
            
            # Convert to CSR format for efficient row operations
            csr_matrix = matrix.tocsr()
            
            # Count the number of non-zero entries in each row
            row_nonzeros = np.diff(csr_matrix.indptr)

            #make a count vector
            count_vector = [0] * 20

            for n in row_nonzeros:
                count_vector[n] += 1
            
            # Append the row counts to the list
            row_conn_append.append(count_vector)
            #get number of cells
            cell_count.append(row_nonzeros.shape[0])
            #extend
            # row_connections.extend(row_nonzeros)
            adj_matrix_list.append(matrix)

        df = pd.DataFrame(row_conn_append, columns=[f'{i}' for i in range(20)])
        df['dataset_id'] = dataset_ids
        df['tissue_type'] = file_list[count]
        df['total_cells'] = cell_count
        # df['dataset_ids'] = dataset_list[0]

        # max_length = max(len(row) for row in row_conn_append)
        # padded_dataset = np.array([np.pad(row, (0, max_length - len(row)), 'constant') for row in row_conn_append])

        # tissue_cell_conn.append(row_connections)
        df_tissue_cell_conn.append(df)
        
        count += 1
        continue


        # get qm data
        # qm_df = qm_process(qm_pathlist, common_channels, pixels, file_list)
        # common_tissue_snr.append(snr_common)

        # save snr_common & tot_l
        # out = output_dir / (file_list[count] + '-snr_common.npy')
        # np.save(out, snr_common)

        # #save total intensity
        # out = output_dir / (file_list[count] + '-total_intensity_common.npy')
        # np.save(out, tot_l)

        # #save quality score
        # qs_df.to_pickle(output_dir / (file_list[count] + '-qs.pkl'))

        # #save cellbg
        # out = output_dir / (file_list[count] + '-cellbg.npy')
        # np.save(out, cellbg_l)

        #save qm_df
        # out = output_dir / (file_list[count] + 'qm_df.csv')
        # qm_df.to_csv(out, index=False)

        # count += 1
        # continue

        seg_list = [mean_pathlist, covar_pathlist, total_pathlist]

        mean_paths = feature_seg(mean_pathlist)
        covar_paths = feature_seg(covar_pathlist)
        total_paths = feature_seg(total_pathlist)

        seg_list = [total_paths, covar_paths]
        # seg_list = [mean_paths, covar_paths]


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
        shape_matrix = pd.concat([pd.read_csv(t) for t in shape_pathlist], ignore_index=True)

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
        total_matrix_cells = total_matrix[0]
        covar_matrix_cells = covar_matrix[1]
        mean_matrix_cells = mean_matrix[0]

        total_matrix = total_matrix.reshape(
            (total_matrix.shape[0], total_matrix.shape[1], total_matrix[2] * total_matrix[3]))
        total_matrix = pd.concat(total_matrix[1:], axis=1)



        # recreate mean all
        mean_all_matrix = np.concatenate(mean_matrix, axis=1)

        # save down sample of all features for local processing
        all_feats = [covar_matrix, total_matrix, mean_all_matrix, shape_matrix, total_matrix_cells]
        # all_feats = [covar_matrix_cells, mean_matrix_cells]

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

    #concat df 
    df_full = pd.concat(df_tissue_cell_conn)

    # labels = []
    # for i, dataset in enumerate(tissue_cell_conn):
    #     labels.extend([file_list[i]] * len(dataset)) 
    # labels = np.concatenate([[i] * len(padded_datasets[i]) for i in range(len(padded_datasets))])

    

    # reducer_pca = PCA(n_components=2, random_state=42)
    reducer_umap = umap.UMAP(n_components=2, random_state=42)

    df_drop = df_full.drop(columns=['tissue_type', 'dataset_id'])

    #normalize
    df_drop = df_drop.apply(lambda x: x / x['total_cells'], axis=1)

    df_drop = df_drop.drop(columns=['total_cells'])

    # pca_reduced_data = reducer_pca.fit_transform(df_drop)
    umap_reduced_data = reducer_umap.fit_transform(df_drop)


    #plot histogram of tissue cell conn 
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

    # Create the plot
    # plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))


    # Plot each point with color corresponding to its label
    # Plot PCA results
    # ax = axes[0]
    # for i in range(len(file_list)):
    #     # idx = labels == i
    #     idx = df_full['tissue_type'] == file_list[i]
    #     ax.scatter(pca_reduced_data[idx, 0], pca_reduced_data[idx, 1], color=colors[i], label=file_list[i], alpha=0.5)
    # # ax.set_title('PCA Result')
    # # ax.set_xlabel('Component 1')
    # # ax.set_ylabel('Component 2')
    # ax.set_title('A', fontsize=20, fontweight='bold')
    # ax.legend()

    # Plot UMAP results
    ax = axes[0,0]
    for i in range(len(file_list)):
        idx = df_full['tissue_type'] == file_list[i]
        ax.scatter(umap_reduced_data[idx, 0], umap_reduced_data[idx, 1], color=colors[i], label=file_list[i], alpha=0.5)
    # ax.set_title('UMAP Result')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('A', fontsize=20, fontweight='bold')
    ax.legend()
    # ax.set_aspect('equal')

    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.legend()
    # fig.text(0.5, 0.04, 'Component 1', ha='center', va='center', fontsize=20)
    # fig.text(0.04, 0.5, 'Component 2', ha='center', va='center', rotation='vertical', fontsize=20)
    # Save the plot to a file
    # plt.savefig('pca_umap_plot.png', dpi=300)

    # df_full['pc1'] = pca_reduced_data[:, 0]
    # df_full['pc2'] = pca_reduced_data[:, 1]
    df_full['umap1'] = umap_reduced_data[:, 0]
    df_full['umap2'] = umap_reduced_data[:, 1]

    df_full.to_csv('df_adj.csv', index=False)

    kmeans_model = KMeans(n_clusters=4, random_state=42)
    kmeans_model.fit(umap_reduced_data)

    centroids = kmeans_model.cluster_centers_
    closest_indices, _ = pairwise_distances_argmin_min(centroids, umap_reduced_data)

    # Predict cluster labels for the data points
    labels = kmeans_model.labels_

    # Define colors for each cluster
    colors = ['blue', 'green', 'red', 'cyan']
    markers = ['3', '1', '4', '2']
    symbols = ["X", "D", "H", "*"]

    # Plot the UMAP-reduced data with cluster labels
    # plt.figure(figsize=(10, 8))
    # for i in range(4):
    #     cluster_data = umap_reduced_data[labels == i]
    #     plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=colors[i], label=f'Cluster {i+1}', alpha=0.5)

    # centroid_subset = umap_reduced_data[closest_indices]

    # Plot the closest point to centroid
    # plt.scatter(umap_reduced_data[closest_indices, 0], umap_reduced_data[closest_indices, 1], color='black', marker='x', s=100, label='Centroids')
    # Plot the centroids with distinct markers
    for idx, marker, symbol in zip(closest_indices, markers, symbols):
        ax.scatter(umap_reduced_data[idx, 0], umap_reduced_data[idx, 1], color='black', marker=symbol, s=50)  # Plot the point
        ax.text(umap_reduced_data[idx, 0], umap_reduced_data[idx, 1], marker, fontsize=16, ha='right', color='black')  # Add text label

    # ax.set_title('A', fontsize=20, fontweight='bold')

    # Add title and labels
    # plt.title('UMAP Clusters and Centroids')
    # plt.xlabel('UMAP Component 1')
    # plt.ylabel('UMAP Component 2')
    # plt.legend()
    # plt.savefig('umap_centroids.png', dpi=300)

    # plt.clf()
    # plt.close()

    # plt.figure(figsize=(10, 8))
    ax = axes[0,1]
    #plot line graph 
    centroid_subset = df_drop.iloc[closest_indices]
    x = np.arange(13)
    for i, (index, row) in enumerate(centroid_subset.iterrows()):
        ax.plot(x, row.iloc[:len(x)], label=markers[i], marker=symbols[i])
        
    ax.set_xlabel('Degree of Nodes')
    ax.set_ylabel('Fraction of Cells')
    ax.legend()
    ax.set_title('B', fontsize=20, fontweight='bold')
    # plt.savefig('umap_centroid_plot.png', dpi=300)

    # 72 and 90 idx
    ax = axes[1, 0]
    ax.set_title('C', fontsize=20, fontweight='bold')

    mask_path = Path('/hive/hubmap/data/consortium/Stanford TMC/0511b3adf57a06d72d5ebe2204d076fe/stitched/mask/reg1_stitched_mask.ome.tiff')
    mask = tifffile.imread(mask_path)[0][0]

    matrix = adj_matrix_list[90]

    # Convert to CSR format for efficient row operations
    csr_matrix = matrix.tocsr()

    # Count the number of non-zero entries in each row
    row_nonzeros = np.diff(csr_matrix.indptr)

    cell_center_df = cell_center_list[90].iloc[1:].reset_index()
    cell_center_df['connections'] = row_nonzeros

    unique_connections = cell_center_df['connections'].unique()
    # color_map = {conn: i for i, conn in enumerate(unique_connections)}

    # Create a color array
    # colors = [color_map[conn] for conn in cell_center_df['connections']]

    cmap = plt.get_cmap('tab10')

    n_colors = 11 # At least 11 colors (0-10)
    colors = ['grey' if i == 0 else cmap(i % 10) for i in range(n_colors)]
    cmap = ListedColormap(colors)

    # Create a normalization object
    # norm = BoundaryNorm(np.arange(-0.5, 10, 1), cmap.N)
    norm = BoundaryNorm(np.arange(-0.5, n_colors + 0.5, 1), cmap.N)

    # ax.scatter(cell_center_df['x'], cell_center_df['y'], c=colors, cmap='tab20', marker=",")
    for i in unique_connections: 
        matching_indices = (cell_center_df.index[cell_center_df['connections'] == i] + 1).tolist()
        mask_bool = np.isin(mask, matching_indices)
        mask[mask_bool] = i
    
    #map greater than 10 to 10
    mask = np.minimum(mask, 10)
    # unique_values = np.unique(mask)
    
    im = ax.imshow(mask, cmap=cmap, norm=norm)
    

    # # Find non-zero entries and create lines

    # for i in range(csr_matrix.shape[0]):
    #     row = csr_matrix.getrow(i)
    #     non_zero_cols = row.nonzero()[1]
    #     lines = []
    #     line_colors = []
    #     for j in non_zero_cols:
    #         if i != j:  # Avoid self-connections if they exist
    #             start = cell_center_df.loc[i, ['x', 'y']].values
    #             end = cell_center_df.loc[j, ['x', 'y']].values
    #             lines.append([start, end])
    #             # Use the color of the start point for the line
    #             line_colors.append(cmap(colors[i]))

    #     lc = LineCollection(lines, colors=line_colors, alpha=1, linewidths=1)
    #     ax.add_collection(lc)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')

    # ########

    ax = axes[1, 1]
    ax.set_title('D', fontsize=20, fontweight='bold')

    mask_path = Path('/hive/hubmap/data/consortium/Stanford TMC/76e46e1fe30f3a911d3cf5c1e4530351/pipeline_output/mask/reg001_mask.ome.tiff')
    mask = tifffile.imread(mask_path)[0][0]

    matrix = adj_matrix_list[72]

    # Convert to CSR format for efficient row operations
    csr_matrix = matrix.tocsr()

    # Count the number of non-zero entries in each row
    row_nonzeros = np.diff(csr_matrix.indptr)

    cell_center_df = cell_center_list[72].iloc[1:].reset_index()
    cell_center_df['connections'] = row_nonzeros

    unique_connections = cell_center_df['connections'].unique()
    # color_map = {conn: i for i, conn in enumerate(unique_connections)}

    # # Create a color array
    # colors = [color_map[conn] for conn in cell_center_df['connections']]

    # ax.scatter(cell_center_df['x'], cell_center_df['y'], c=colors, cmap='tab20', marker=',')

    # Create a normalization object
    # norm = BoundaryNorm(np.arange(-0.5, 10, 1), cmap.N)

    # ax.scatter(cell_center_df['x'], cell_center_df['y'], c=colors, cmap='tab20', marker=",")
    for i in unique_connections: 
        matching_indices = (cell_center_df.index[cell_center_df['connections'] == i] + 1).tolist()
        mask_bool = np.isin(mask, matching_indices)
        mask[mask_bool] = i
    
    #map greater than 10 to 10
    mask = np.minimum(mask, 10)
    
    im = ax.imshow(mask, cmap=cmap, norm=norm)
    unique_values = np.unique(mask)

    # Create custom legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap(norm(value)), 
                                    edgecolor='none', label=str(value) if value < 10 else '10+')
                    for value in unique_values]
    ax.legend(handles=legend_elements, title='Legend', loc='lower left', 
            bbox_to_anchor=(0, 0))

    # Add colorbar
    # cbar = plt.colorbar(im, ax=ax, ticks=unique_values)
    # cbar.set_label('Values')

    # # Create custom legend
    # legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap(norm(value)), 
    #                                 edgecolor='none', label=str(value) if value < 10 else '10+')
    #                 for value in unique_values]
    # ax.legend(handles=legend_elements, title='Legend', loc='center left', 
    #         bbox_to_anchor=(1, 0.5))


    # # Get the color map
    # cmap = plt.get_cmap('tab20')

    # # Find non-zero entries and create lines

    # for i in range(csr_matrix.shape[0]):
    #     row = csr_matrix.getrow(i)
    #     non_zero_cols = row.nonzero()[1]
    #     lines = []
    #     line_colors = []
    #     for j in non_zero_cols:
    #         if i != j:  # Avoid self-connections if they exist
    #             start = cell_center_df.loc[i, ['x', 'y']].values
    #             end = cell_center_df.loc[j, ['x', 'y']].values
    #             lines.append([start, end])
    #             # Use the color of the start point for the line
    #             line_colors.append(cmap(colors[i]))

    #     lc = LineCollection(lines, colors=line_colors)
    #     ax.add_collection(lc)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')

    
    plt.savefig('new_figure.png', dpi=1000)
    plt.clf()
    plt.close()

    exit()

    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 20))

    # axs = axes2[0, 0] 
    # axs.set_title('1', fontsize=20, fontweight='bold')

    mask = cell_connection_mask('/hive/hubmap/data/consortium/Stanford TMC/0511b3adf57a06d72d5ebe2204d076fe/stitched/mask/reg1_stitched_mask.ome.tiff', 
                                adj_matrix_list[90], cell_center_list[90])
    
    mask2 = cell_connection_mask('/hive/hubmap/data/consortium/Stanford TMC/76e46e1fe30f3a911d3cf5c1e4530351/pipeline_output/mask/reg001_mask.ome.tiff',
                                adj_matrix_list[72], cell_center_list[72])
    
    mask3 = cell_connection_mask('/hive/hubmap/data/consortium/Stanford TMC/35454f56fd4d7b3c4b80921a9c7e6b18/pipeline_output/mask/reg001_mask.ome.tiff',
                                adj_matrix_list[78], cell_center_list[78])
    
    #spleen
    mask4 = cell_connection_mask('/hive/hubmap/data/public/19184e64b152cd9977f56785da9495fd/stitched/mask/reg1_stitched_mask.ome.tiff',
                                adj_matrix_list[25], cell_center_list[25])

    mask5 = cell_connection_mask('/hive/hubmap/data/public/666d5c8292a7ef2e7cddc9fd22b1e0df/stitched/mask/reg1_stitched_mask.ome.tiff',
                                adj_matrix_list[18], cell_center_list[18])

    mask6 = cell_connection_mask('/hive/hubmap/data/public/3bf7c7b6444c3c3d6dbaf543e85cfc2b/stitched/mask/reg1_stitched_mask.ome.tiff',
                                 adj_matrix_list[28], cell_center_list[28])

    unique_values = np.unique(mask2)

    # Create custom legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap(norm(value)), 
                                    edgecolor='none', label=str(value) if value < 10 else '10+')
                    for value in unique_values]

    labels2 = ['1', '2', '3', '2', '3', '4']
    masks = [mask, mask2, mask3, mask4, mask5, mask6]
    count = 0
    for i in range(0, 2):
        for j in range(0, 3):
            axes2[i, j].set_title(labels2[count], fontsize=20, fontweight='bold')
            axes2[i, j].imshow(masks[count], cmap=cmap, norm=norm)

            count += 1

    axes2[1, 2].legend(handles=legend_elements, title='Legend', loc='lower left', 
            bbox_to_anchor=(0, 0))

    plt.savefig('new_figure2.png', dpi=1000)
    



    # for i, cell_coord in cell_center_list[90].iterrows():
    #     if i == 0:
    #         continue

    #     if idx and all(x in cells for x in idx):
    #         neighbors = cell_center.loc[idx, :]
    #         lines = []
    #         for j, neighbor_coord in neighbors.iterrows():
    #             lines.append([cell_coord, neighbor_coord])

    #             # dist = adjacencyMatrix[i, j]
    #             # gap = (neighbor_coord - cell_coord) / 2
    #             # ax.text(
    #             #     cell_coord[0] + gap[0],
    #             #     cell_coord[1] + gap[1],
    #             #     '%.1f' % dist,
    #             #     ha='center',
    #             #     va='center',
    #             #     fontsize='xx-small'
    #             # )
    #         line = mc.LineCollection(lines, colors=[(1, 0, 0, 1)])
    #         ax.add_collection(line)









    exit()

    # Plot each row_connections as a histogram
    # for i, row_connections in enumerate(tissue_cell_conn):
    #     plt.hist(row_connections, bins='auto', alpha=0.5, color=colors[i % len(colors)], label=file_list[i])
    plt.hist(tissue_cell_conn, label=file_list)

    plt.xlabel('Number of Connected Cells')
    plt.ylabel('Frequency')
    # plt.title('Histogram of Connections per Row in Sparse Matrices')
    plt.legend()

    plt.savefig(output_dir / 'histogram_conn_cells.png', dpi=300)

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
    # plt.hist(qsl, label=file_list)
    # plt.legend()
    # plt.title('Segmentation Quality Score Among Different Tissues')
    # plt.ylabel('Frequency')
    # plt.xlabel('Quality Score')
    # plt.savefig(output_dir / 'qs_tissues.png', bbox_inches='tight')
    # plt.clf()

    # nCpsm
    # plt.hist(nCpsm_l, label=file_list)
    # plt.legend()
    # plt.title('Number of Cells Per 100 Square Microns Among Different Tissues')
    # plt.ylabel('Frequency')
    # plt.xlabel('Cells per 100 microns squared')
    # plt.savefig(output_dir / 'nCpsm.png', bbox_inches='tight')

def cell_connection_mask(mask_path: Path, matrix: list, cell_center_df: pd.DataFrame):

    mask_path = Path(mask_path)
    mask = tifffile.imread(mask_path)[0][0]

    # matrix = adj_matrix_list[28]

    # Convert to CSR format for efficient row operations
    csr_matrix = matrix.tocsr()

    # Count the number of non-zero entries in each row
    row_nonzeros = np.diff(csr_matrix.indptr)

    cell_center_df = cell_center_df.iloc[1:].reset_index()
    cell_center_df['connections'] = row_nonzeros

    unique_connections = cell_center_df['connections'].unique()

    for i in unique_connections: 
        matching_indices = (cell_center_df.index[cell_center_df['connections'] == i] + 1).tolist()
        mask_bool = np.isin(mask, matching_indices)
        mask[mask_bool] = i
    
    #map greater than 10 to 10
    mask = np.minimum(mask, 10)

    return mask

def AdjacencyMatrix2Graph(adjacencyMatrix, cell_center: np.ndarray, cellGraph, name, thr):
    cell_center = pd.DataFrame(cell_center)
    cells = set(cell_center.index)
    fig, ax = plt.subplots(figsize=(17.0, 17.0))
    plt.plot(cell_center.iloc[:, 0], cell_center.iloc[:, 1], ",")
    plt.title("Cell Adjacency Graph, distance <" + str(thr))
    for i, cell_coord in cell_center.iterrows():
        idx = list(cellGraph[i])
        if idx and all(x in cells for x in idx):
            neighbors = cell_center.loc[idx, :]
            lines = []
            for j, neighbor_coord in neighbors.iterrows():
                lines.append([cell_coord, neighbor_coord])

                # dist = adjacencyMatrix[i, j]
                # gap = (neighbor_coord - cell_coord) / 2
                # ax.text(
                #     cell_coord[0] + gap[0],
                #     cell_coord[1] + gap[1],
                #     '%.1f' % dist,
                #     ha='center',
                #     va='center',
                #     fontsize='xx-small'
                # )
            line = mc.LineCollection(lines, colors=[(1, 0, 0, 1)])
            ax.add_collection(line)
    plt.savefig(name, **figure_save_params)
    plt.clf()
    plt.close()



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


def qm_process(paths, common_channels, tissue_pixels, file_list):
    # dict_l = []
    # # qsl = []
    # # alldf = []
    # # nCpsm_l = []
    # tot_l = []
    # cellbg_l = []

    qm_data = []

    for count, i in enumerate(paths):
        # qm = json.load(i.read_bytes())
        with open(i) as f:
            qm = json.load(f)

        img_id = i.parent.name

        s2n_otsu = qm['Image Quality Metrics not requiring image segmentation']['Signal To Noise Otsu']
        s2n_z = qm['Image Quality Metrics not requiring image segmentation']['Signal To Noise Z-Score']
        qs = qm['Segmentation Evaluation Metrics']['QualityScore']
        # nCpsm = qm['Segmentation Evaluation Metrics']['Matched Cell']['NumberOfCellsPer100SquareMicrons']

        # for quality metrics
        # nChannels = qm['Image Information']['Number of Channels']
        # invAvgcbg = qm['Image Quality Metrics requiring background segmentation']['1/AvgCVBackground']
        fracPBG = qm['Image Quality Metrics requiring background segmentation']['Fraction of Pixels in Image Background']
        fracImgCells = qm['Image Quality Metrics that require cell segmentation']['Fraction of Image Occupied by Cells']
        # nCells = qm['Image Quality Metrics that require cell segmentation']['Number of Cells']

        # iterate through keys
        # cBG = 0
        # nucCell = 0
        avgCellR = qm['Image Quality Metrics that require cell segmentation']['Channel Statistics'][
            'Average per Cell Ratios']
        # for key in avgCellR.values():
            # cBG += key['Cell / Background']
            # nucCell += key['Nuclear / Cell']
        cellbg = [value['Cell / Background'] for key, value in avgCellR.items() if key in common_channels]
        # cellbg_l.append(cellbg)
        # avg
        # avgcBG = cBG / nChannels
        # avgnucCell = nucCell / nChannels

        # append to df
        # to_append = [fracImgCells, nCpsm, qs, fracPBG]
        # series_append = pd.Series(to_append, index=df.columns)
        # df = df.append(series_append, ignore_index=True)

        # common channel names
        common_otsu = [s2n_otsu['CD11c'], s2n_otsu['CD21'], s2n_otsu['CD4'], s2n_otsu['CD8'], s2n_otsu['Ki67']]
        common_z = [s2n_z['CD11c'], s2n_z['CD21'], s2n_z['CD4'], s2n_z['CD8'], s2n_z['Ki67']]
        # a = np.asarray(common_otsu)
        # b = np.asarray(common_z)
        #
        # c = np.concatenate((a, b), axis=0)
        # dict_l.append(c)
        # dict_l = np.asarray(dict_l)

        # qsl.append(qs)

        # total intensity of common channels
        meanInt1 = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']['CD11c'] / tissue_pixels[count]
        meanInt2 = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']['CD21'] / tissue_pixels[count]
        meanInt3 = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']['CD4'] / tissue_pixels[count]
        meanInt4 = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']['CD8'] / tissue_pixels[count]
        meanInt5 = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']['Ki67'] / tissue_pixels[count]

        all_tot = [meanInt1, meanInt2, meanInt3, meanInt4, meanInt5]
        # tot_l.append(all_tot)

        # a = np.asarray(list(s2n_otsu.values()))
        # b = np.asarray(list(s2n_z.values()))
        # c = (a + b) / 2
        # c = c.reshape(1, len(c))
        #
        # df_2 = pd.DataFrame(c, columns=s2n_otsu.keys())
        #
        # alldf.append(df_2)
        # nCpsm_l.append(nCpsm)

        #qm list
        qm_list = [img_id, fracImgCells, fracPBG, qs]

    # cellbg = np.asarray(cellbg_l)
    # dict_l = np.asarray(dict_l)
    # tot_l = np.asarray(tot_l)


        qm_data_row = np.concatenate((qm_list, common_otsu, common_z, all_tot, cellbg))
        qm_data.append(qm_data_row)

    qm_data = np.asarray(qm_data)
    qm_df = pd.DataFrame(qm_data, columns=['ImgID', 'FracImgOfCells', 'FracPixInImgBG', 'SegQS', 
                                           'Otsu: CD11c', 'Otsu: CD21', 'Otsu: CD4', 'Otsu: CD8', 'Otsu: Ki67',
                                           'Z-Score: CD11c', 'Z-Score: CD21', 'Z-Score: CD4', 'Z-Score: CD8', 'Z-Score: Ki67', 
                                           'meanInt: CD11c', 'meanInt: CD21', 'meanInt: CD4', 'meanInt: CD8', 'meanInt: Ki67',
                                            'Cell/BG: CD11c', 'Cell/BG: CD21', 'Cell/BG: CD4', 'Cell/BG: CD8', 'Cell/BG: Ki67'
    ])

    qm_df['tissue'] = file_list[0]


    return qm_df 

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


def box_n_whisker(tissue_snr, file_list, output_dir):

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
