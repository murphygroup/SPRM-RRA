import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
from ast import literal_eval
from scipy import interpolate, stats
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.ndimage as ndimage
import cv2


def main():
    # root = Path('/hive/hubmap/lz/tedz-share/HUBMAP_DATA/')
    # root = Path('/Users/tedzhang/Desktop/murphylab/hubmap/github/SPRM/sprm_outputs_oldish/')
    # root = Path('/home/tedz/Downloads/sprm-analysis/inputs/')
    root = Path('/Users/tedzhang/Desktop/CMU/hubmap/SPRM/MANUSCRIPT/input')
    # florida = root / 'Florida'
    # stanford = root / 'Stanford'
    #
    # lymphnodes = florida / 'LN'
    # spleen = florida / 'SPLEEN'
    # thymus = florida / 'THYMUS'
    # largeint = stanford / 'LI'
    # smallint = stanford / 'SI'

    # tissue_list = [lymphnodes, spleen, thymus, largeint, smallint]
    file_list = ['lymph_nodes', 'spleen', 'thymus', 'large_intestine', 'small_intestine']
    flat_l = []
    subset_l = []
    skipped_cells = []
    output_dir = Path('/Users/tedzhang/Desktop/CMU/hubmap/SPRM/MANUSCRIPT/output')
    # output_dir = Path.home() / 'analysis' / 'outputs'
    # random init
    rng = np.random.default_rng(42)

    #init
    # count = 0

    largeint_cp = np.load((root / 'large intestine-cell_polygons.npy'), allow_pickle=True)
    lymphn_cp = np.load(root / 'lymph nodes-cell_polygons.npy', allow_pickle=True)
    smallint_cp = np.load(root / 'small intestine-cell_polygons.npy', allow_pickle=True)
    spleen_cp = np.load(root / 'spleen-cell_polygons.npy', allow_pickle=True)
    thymus_cp = np.load(root / 'thymus-cell_polygons.npy', allow_pickle=True)

    all_l = [lymphn_cp, spleen_cp, thymus_cp, largeint_cp, smallint_cp]
    #subsample cells
    step = 500
    rndperm = rng.permutation(largeint_cp.shape[0]) #all should have the same amount of cells
    rndperm_sublists = [rndperm[i:i + step] for i in range(0, len(rndperm), step)]
    subsample_matrix_l = [x[idx] for (x, idx) in zip(all_l, rndperm_sublists)]


    all_l = np.concatenate(subsample_matrix_l)

    normalized_cells = []
    sf_l = []
    # convert string to floats
    i = 0
    for j in all_l:
        # if i == 4826:
        #     print('here')
        roi = np.asarray(literal_eval(j[0]))
        norm_cell, sf = outline_norm(roi, 100, skipped_cells, i)

        if norm_cell is None:
            i += 1
            continue


        normalized_cells.append(norm_cell)
        sf_l.append(sf)
        i += 1

    # dims: num of cells x 100 (npoints) x 2 (x & y)
    normalized_cells = np.asarray(normalized_cells)
    flat_nc = normalized_cells.reshape((normalized_cells.shape[0], 200))
    flat_l.append(flat_nc)
    new_flat_l = np.asarray(flat_l)
    new_flat_l = np.squeeze(new_flat_l)

    # help out memory
    num_of_cells = normalized_cells.shape[0]
    normalized_cells = None
    flat_nc = None

    # tissue id
    tissue_id = np.zeros((num_of_cells, 1))
    for i in range(len(file_list)):
        start = int(i * num_of_cells / len(file_list))
        end = int((i + 1) * num_of_cells / len(file_list))
        tissue_id[start:end, 0] = i

    # sf
    sf_l = np.asarray(sf_l)[:, np.newaxis]
    # sf_l = np.hstack((id, sf_l))
    # out = root / (file_list[count] + '-scale-factor.npy')
    # # out = output_dir / (file_list[count] + '-scale-factor.npy')
    # np.save(out, sf_l)

    new_flat_l = np.concatenate((tissue_id, sf_l, new_flat_l), axis=1)
    # save it
    # out = root / (file_list[count] + '-normalized-cells.npy')
    out = output_dir / 'normalized-cells.npy'
    np.save(out, new_flat_l)

    # subsample
    # N = 10000
    # rndperm = rng.permutation(num_of_cells)
    # subset_matrix = new_flat_l[rndperm[:N], :].copy()
    # subset_l.append(subset_matrix)
    # print('finished processing randomly sampled submatrix')

    # concatenate
    # subset = np.concatenate(subset_l, axis=0)
    subset_tissue_id = new_flat_l[:, 0].astype(int)
    subset_all = new_flat_l[:, 1:]
    subset_all_no_size = subset_all[:, 1:]

    #z-score don't need since points should already be centered

    subset_all_og = subset_all.copy()
    subset_all_no_size_og = subset_all_no_size.copy()

    pca_l = [subset_all, subset_all_no_size]
    pca_og = [subset_all_og, subset_all_no_size_og]
    pca_results = []
    pca_models = []
    sil_list = []

    names = ['size', 'shape']
    for i in range(len(pca_l)):
        while True:
            try:
                m = PCA(n_components=100).fit(pca_l[i])
                break
            except Exception as e:
                print(e)
                n_samples = int(pca_l[i].shape[0] / 2)
                idx = np.random.choice(
                    pca_og[i].shape[0], n_samples, replace=False
                )
                pca_l[i] = pca_og[i][idx, :]

        # cluster on the # of PCs for j in range(2, 6): num_cluster, kmeans_labels, cluster_centers =
        # get_silhouette_score(pca_results[i], 'PCs-outlines-cluster-silhouette-scores-' + str(i), root)
        # sil_list.append(get_silhouette_score(pca_l[i], 'PCs-outlines-cluster-silhouette-scores-' + names[i], output_dir))
        # 1 - num_cluster
        # 2 - kmeans labels
        # 3 - cluster centers

        pca_results.append(m.transform(pca_og[i]))  # num cells x pcs
        pca_models.append(m)

    #find shapes

    shape = pca_results[1]
    ul_shapes = np.where((shape[:, 0] < 1) & (shape[:, 1] > 4))
    ll_shapes = np.where((shape[:, 0] < -1.5) & (shape[:, 1] < 2))
    m_shapes = np.where((shape[:, 0] < 1) & (shape[:, 0] > -1) & (shape[:, 1] < 2))
    b_shapes = np.where((shape[:, 0] < 2) & (shape[:, 0] > 1))
    rm_shapes = np.where((shape[:, 0] > 2) & (shape[:, 1] < 1))
    lr_shapes = np.where((shape[:, 0] > 6) & (shape[:, 1] < 1))
    mr_shapes = np.where((shape[:, 0] > 6) & (shape[:, 1] > 2) & (shape[:, 1] < 4))
    ur_shapes = np.where((shape[:, 0] > 6) & (shape[:, 1] > 6))


    #get the points you want to show
    ul_shapes_pt = [(pca_models[1].inverse_transform(shape[i, :]), i) for i in ul_shapes[0]] # 4 and 5
    ll_shapes_pt = [(pca_models[1].inverse_transform(shape[i, :]), i) for i in ll_shapes[0]]
    m_shapes_pt = [(pca_models[1].inverse_transform(shape[i, :]), i) for i in m_shapes[0]]
    b_shapes_pt = [(pca_models[1].inverse_transform(shape[i, :]), i) for i in b_shapes[0]]
    rm_shapes_pt = [(pca_models[1].inverse_transform(shape[i, :]), i) for i in rm_shapes[0]]
    lr_shapes_pt = [(pca_models[1].inverse_transform(shape[i, :]), i) for i in lr_shapes[0]]
    mr_shapes_pt = [(pca_models[1].inverse_transform(shape[i, :]), i) for i in mr_shapes[0]]
    ur_shapes_pt = [(pca_models[1].inverse_transform(shape[i, :]), i) for i in ur_shapes[0]]

    shape_plots = [ul_shapes_pt[4], ul_shapes_pt[1], ul_shapes_pt[5], ll_shapes_pt[0], m_shapes_pt[0], m_shapes_pt[42], b_shapes_pt[0], rm_shapes_pt[10], rm_shapes_pt[2], mr_shapes_pt[2], mr_shapes_pt[0], ur_shapes_pt[0]]

    #explained variance
    exvar = [m.explained_variance_ratio_ for m in pca_models]
    exvar_l = [["{0:.0%}".format(exvar[0]), "{0:.0%}".format(exvar[1])] for exvar in exvar]

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # label by tissue
    for j in range(len(file_list)):
        idx = np.where(subset_tissue_id == j)
        # rndperm = rng.permutation(idx)
        # idx = rndperm[:step]
        axs[0].scatter(pca_results[0][idx, 0], pca_results[0][idx, 1], label=file_list[j], marker='.', alpha=0.3)
        axs[1].scatter(pca_results[1][idx, 0], pca_results[1][idx, 1], label=file_list[j], marker='.', alpha=0.3)
    # if i == 0:
    # axs[0].legend(loc='center', bbox_to_anchor=(-0.1, 0.85))
    axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    axs[0].set_xlabel('PC-1: ' + exvar_l[0][0] + ' explained variance')
    axs[0].set_ylabel('PC-2: ' + exvar_l[0][1] + ' explained variance')
    axs[1].set_xlabel('PC-1: ' + exvar_l[1][0] + ' explained variance')
    axs[1].set_ylabel('PC-2: ' + exvar_l[1][1] + ' explained variance')
    # if i == 0:
    axs[0].set_title('A', fontsize=10, fontweight='bold')
    # else:
    axs[1].set_title('B', fontsize=10, fontweight='bold')
    # f = root / (str(i) + '-PCs-clusters-KMEANS.png')
    # f = output_dir / (names[i] + '-PCs_size.png')/
    # plt.savefig(f, format='png')
    # plt.clf()

    # for j in range(len(file_list)):
    #     idx = np.where(subset_tissue_id == j)
    #     # rndperm = rng.permutation(idx)
    #     # idx = rndperm[:step]
    #     axs[1].scatter(pca_results[1][idx, 0], pca_results[1][idx, 1], label=file_list[j], marker='.', alpha=0.3)
    # # if i == 0:
    # # plt.legend(loc='center', bbox_to_anchor=(-0.1, 0.85))
    # axs[1].set_xlabel('PC-1: ' + exvar_l[1][0] + ' explained variance')
    # axs[1].set_ylabel('PC-2: ' + exvar_l[1][1] + ' explained variance')
    # if i == 0:
    # plt.title('PCs of Cell Shape and Size')
    # else:
    # plt.title('PCs of Cell Shape')
    # f = root / (str(i) + '-PCs-clusters-KMEANS.png')

    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']


    for i, label in enumerate(labels):
        axs[0].annotate(label, (pca_results[0][shape_plots[i][1], 0], pca_results[0][shape_plots[i][1], 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', fontsize=15)
        axs[1].annotate(label, (pca_results[1][shape_plots[i][1], 0], pca_results[1][shape_plots[i][1], 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', fontsize=15)

    f = root / 'pca-shape-size.png'
    plt.savefig(f, dpi=300)
    plt.clf()
    plt.close(fig)

    fig, axs = plt.subplots(2, 6, sharey=True, sharex=True, figsize=(4, 8))
    c = 0
    for i in range(2):
        for j in range(6):
            axs[i, j].scatter(shape_plots[c][0][::2], shape_plots[c][0][1::2])
            # axs[i, j].set(adjustable="box", aspect="equal")
            axs[i, j].set_title(labels[c], weight='bold')
            # axs[j, i].axis('off')
            c += 1
    plt.subplots_adjust(hspace=0.5)
    f = root / 'cell-shapes-pc.png'
    plt.savefig(f, dpi=300)






    #
    # # Specify the cluster num
    # num_cluster = 2
    # kmeans_size = KMeans(n_clusters=num_cluster)
    # kmeans_size.fit(pca_l[0])
    # kmeans_labels_size = kmeans_size.labels_
    # cluster_centers_size = kmeans_size.cluster_centers_
    #
    # kmeans_shape = KMeans(n_clusters=num_cluster)
    # kmeans_shape.fit(pca_l[1])
    # kmeans_labels_shape = kmeans_shape.labels_
    # cluster_centers_shape = kmeans_shape.cluster_centers_
    #
    # print('plotting loop: ' + str(i))
    #


    # label by kmeans
    # cluster_ids = np.arange(k).tolist()
    # for j in cluster_ids:
    #     idx = np.where(sil_list[0][1] == j)
    #     plt.scatter(pca_results[i][idx, 0], pca_results[i][idx, 1], label=cluster_ids[j]+1)
    # plt.legend()
    # plt.xlabel('PC-1: ' + exvar_l[0][0] + ' explained variance')
    # plt.ylabel('PC-2: ' + exvar_l[0][1] + ' explained variance')
    # if i == 0:
    #     plt.title('PCs of Cell Shape and Size clustered by KMeans')
    # else:
    #     plt.title('PCs of Cell Shape clustered by Kmeans')
    # # f = root / (str(i) + '-PCs-clusters-KMEANS.png')
    # f = output_dir / (names[i] + '-PCs-clusters-KMEANS.png')
    # plt.savefig(f, format='png')
    # plt.clf()

    # idx_tissue = []
    # for j in range(len(file_list)):
    #     idx_tissue.append(np.where(subset_tissue_id == j))
    #
    # new_file_list = file_list[::-1]


    #draw contours for size and shape
    # Calculate the decision boundary (line that splits the clusters)
    # x_size = np.linspace(min(pca_l[0]), max(pca_l[0]), 1000)
    # y_size = np.linspace(min(pca_l[0]), max(pca_l[0]), 1000)
    # X_size, Y_size = np.meshgrid(x_size, y_size)
    # Z_size = kmeans_size.predict(np.c_[X_size.ravel(), Y_size.ravel()]).reshape(X_size.shape)
    #
    # # Draw the decision boundary
    # axs[0].contour(X_size, Y_size, Z_size, levels=[0.5], colors='red')
    #
    # x_shape = np.linspace(min(pca_l[1][:, 0]), max(pca_l[1][:, 0]), 1000)
    # y_shape = np.linspace(min(pca_l[1][:, 1]), max(pca_l[1][:, 1]), 1000)
    # X_shape, Y_shape = np.meshgrid(x_size, y_size)
    # Z_shape = kmeans_shape.predict(np.c_[X_shape.ravel(), Y_shape.ravel()]).reshape(X_shape.shape)
    #
    # # Draw the decision boundary
    # axs[1].contour(X_shape, Y_shape, Z_shape, levels=[0.5], colors='red')


    # f = output_dir / (names[i] + '-PCs_tissue.png')
    # plt.savefig(f, format='png')
    # plt.clf()

    #step through pca plots
    # bin_pca(pca_results[i], 1, subset, names[i], i, output_dir)

    # find the cell outline that are closest to each respective center
    # returns a list of idx of closest points in relation to each centroid
    # closest, _ = pairwise_distances_argmin_min(cluster_centers_shape, pca_l[1])
    #
    # #shape
    # fig, ax = plt.subplots(1, len(closest), sharex=True, sharey=True)
    # cent = 0
    # # minidx = 1 - i
    # for k in closest:
    #     ax[cent].scatter(pca_l[1][k, ::2], pca_l[1][k, 1::2])
    #     ax[cent].set(adjustable="box", aspect="equal")
    #     ax[cent].set_title("Cluster-ID: " + str(kmeans_labels_shape[k] + 1))
    #     cent += 1
    #
    # fig.suptitle("K-Medoids Cluster Outlines", fontsize=16)
    # fig.tight_layout()
    # f2 = output_dir / (names[i] + "-cluster-centroid-outline.png")
    # # f2 = root / (str(cent) + "-cluster-centroid-outline.png")
    # plt.savefig(f2, format="png")
    # plt.close(fig)

    # pc1_recon = [-4, -2, 0, 2, 4, 6, 8]
    # pc2_recon = [-2, 0, 2, 4, 6, 8, 10]
    # pc_recon = [pc1_recon, pc2_recon]
    # figure_tit = ["i.", "ii.", "iii.", "iv.", "v.", "vi.", "vii."]
    # #reconstruct cell shape by changing PC1
    # # Now, let's modify the transformed data. For instance, let's change the first component to a constant and others to zero.
    # X_pca_modified = np.zeros_like(pca_results[1])
    #
    #
    # fig, axs = plt.subplots(2, len(pc1_recon), sharey=True, sharex=True)
    # for j in range(2):
    #     for i in range(len(pc1_recon)):
    #         if j==1:
    #             X_pca_modified = np.zeros_like(pca_results[1])
    #         X_pca_modified[:, j] = pc_recon[j][i]  # 'constant' is the value you want to set for the first PC
    #         # Use inverse_transform to reconstruct the original data
    #         X_reconstructed = pca_models[1].inverse_transform(X_pca_modified)
    #
    #         axs[j, i].scatter(X_reconstructed[closest[1], ::2], X_reconstructed[closest[1], 1::2])
    #         axs[j, i].set(adjustable="box", aspect="equal")
    #         axs[0, i].set_title(figure_tit[i])
    #         # axs[j, i].axis('off')
    # axs[0, 3].text(0.5, 1.5, 'A', transform=axs[0, 3].transAxes,
    #                va='bottom', ha='center', weight='bold')
    #
    # axs[1, 3].text(0.5, 1.5, 'B', transform=axs[1, 3].transAxes,
    #                va='bottom', ha='center', weight='bold')
    #
    # f3 = output_dir / "pca_reconstruction.png"
    # plt.savefig(f3, format="png")
    # plt.close(fig)

    print('END')




def bin_pca(features, npca, cell_coord, filename, j, output_dir):

    sort_idx = np.argsort(features[:, npca - 1])  # from min to max
    # set the pc1 intercept
    pc2 = np.median(features[:, 1])

    #cell_coord
    if j == 0:
        cell_coord = (cell_coord[:, 2:].T * cell_coord[:, 1]).T
    else:
        cell_coord = cell_coord[:, 2:]

    p0 = np.array([features[sort_idx[0], 0], pc2])
    p1 = np.array([features[sort_idx[-1], 0], pc2])
    pcs = features[:, 0:2]

    idx = list(np.round(np.linspace(0, len(sort_idx) - 1, 11)).astype(int))
    # nfeatures = features[sort_idx, 0]
    cbin = []

    for i in range(10):
        # fbin = nfeatures[idx[i]:idx[i + 1]]
        new_idx = sort_idx[idx[i]: idx[i + 1]]
        fbin = pcs[new_idx, :]

        x = np.cross(p1 - p0, fbin - p1)
        d = x / np.linalg.norm(p1 - p0)
        closest_pts = np.argsort(d)[0]

        cbin.append(new_idx[closest_pts])

        # find median not mode
        # median = np.median(fbin)
        # mode = stats.mode(fbin)

        # nidx = np.searchsorted(fbin, median, side="left")
        # r = range(idx[i], idx[i + 1])
        # celln = sort_idx[r[nidx]]
        # cbin.append(celln)

    # closest_point = pc[np.argmin(np.linalg.norm(np.cross(p1 - p0, p0 - pc, axisb=1), axis=1) / np.linalg.norm(p1 - p0))]

    f, axs = plt.subplots(1, 10, sharex=True, sharey=True, figsize=(15, 2))

    for i in range(10):
        # cscell_coords = np.column_stack(cell_coord[cbin[i+1]])
        axs[i].scatter(cell_coord[cbin[i], 0::2], cell_coord[cbin[i], 1::2], marker=".")
        axs[i].set_aspect("equal", adjustable="box")

    if j != 0:
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
    else:
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)

    plt.subplots_adjust(wspace=0.75)
    # plt.show()
    plt.savefig(output_dir / (filename + "-outlinePCA_bin_pca.png"))
    plt.close(f)

def outline_norm(ROI, npoints, skipped_cells, cell_idx):
    # make a filled cell
    xmin = min(ROI[:, 0])
    ymin = min(ROI[:, 1])

    x = ROI[:, 0] - xmin
    y = ROI[:, 1] - ymin

    x = x.astype(int)
    y = y.astype(int)

    cmask = np.zeros((max(x) + 1, max(y) + 1))
    cmask[x, y] = 1
    cmask = ndimage.binary_fill_holes(cmask).astype(int)
    coords = np.where(cmask == 1)

    #skip outlines that are not closed
    if len(coords[0]) <= len(x):
        skipped_cells.append(cell_idx)
        return None, None

    # potential change
    ############
    # contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnt1 = contours[0]
    # cnt = cv2.convexHull(contours[0])
    # angle = cv2.minAreaRect(cnt)[-1]
    # print("Actual angle is:" + str(angle))
    # rect = cv2.minAreaRect(cnt)
    #
    # p = np.array(rect[1])
    #
    # if p[0] < p[1]:
    #     print("Angle along the longer side:" + str(rect[-1] + 180))
    #     act_angle = rect[-1] + 180
    # else:
    #     print("Angle along the longer side:" + str(rect[-1] + 90))
    #     act_angle = rect[-1] + 90
    # # act_angle gives the angle of the minAreaRect with the vertical
    #
    # if act_angle < 90:
    #     angle = (90 + angle)
    #     print("angleless than -45")
    #
    #     # otherwise, just take the inverse of the angle to make
    #     # it positive
    # else:
    #     angle = act_angle - 180
    #     print("grter than 90")
    #
    # # rotate the image to deskew it
    # (h, w) = image.shape[:2]
    # print(h, w)
    # center = (w // 2, h // 2)
    # print(center)
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    ############
    x = coords[0] + xmin
    y = coords[1] + ymin

    ptsx = x - round(np.mean(x))
    ptsy = y - round(np.mean(y))
    ptscentered = np.stack([ptsx, ptsy])

    xmin = min(ptscentered[0, :])
    ymin = min(ptscentered[1, :])

    xxx = ptscentered[0, :] - xmin
    yyy = ptscentered[1, :] - ymin
    xxx = xxx.astype(int)
    yyy = yyy.astype(int)

    xmax = max(xxx) + 1
    ymax = max(yyy) + 1

    cmask = np.zeros((xmax, ymax))
    cmask[xxx, yyy] = 1

    ptscov = np.cov(ptscentered)

    eigenvals, eigenvecs = np.linalg.eig(ptscov)
    # print(eigenvals,eigenvecs)
    sindices = np.argsort(eigenvals)[::-1]
    # print(sindices)
    x_v1, y_v1 = eigenvecs[:, sindices[0]]  # eigenvector with largest eigenvalue
    theta = np.arctan((x_v1) / (y_v1))

    rotationmatrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    tmat = rotationmatrix @ ptscentered
    xrotated, yrotated = np.asarray(tmat)
    # plt.plot(xrotated,yrotated,'b+')
    # plt.show()
    # need to flip over minor axis if necessary

    tminx = min(xrotated)
    # print(tminx)
    tminy = min(yrotated)
    # print(tminy)
    xrotated = xrotated - tminx
    #        print(xrotated)
    tmatx = xrotated.round().astype(int)
    #        print(tmatx)
    yrotated = yrotated - tminy
    tmaty = yrotated.round().astype(int)

    # check skew
    x = stats.skew(tmatx)
    y = stats.skew(tmaty)

    # 'heavy' end is on the left side flip to right - flipping over y axis
    if x > 0:
        tmatx = max(tmatx) - tmatx
    elif y > 0:
        tmaty = max(tmaty) - tmaty
    elif x > 0 and y > 0:
        tmatx = max(tmatx) - tmatx
        tmaty = max(tmaty) - tmaty

    # make the object mask have a border of zeroes
    cmask = np.zeros((max(tmatx) + 3, max(tmaty) + 3))
    cmask[tmatx + 1, tmaty + 1] = 1
    # fill the image to handle artifacts from rotation
    # cmask = fillimage(cmask)
    cmask = ndimage.binary_fill_holes(cmask).astype(int)

    # remove isolated pixels
    cmask = remove_island_pixels(cmask)

    # plt.imshow(cmask)
    # plt.show()

    aligned_outline = measure.find_contours(
        cmask, 0.5, fully_connected="high", positive_orientation="low"
    )

    x = aligned_outline[0][:, 0] + tminx
    y = aligned_outline[0][:, 1] + tminy

    x, y = linear_interpolation(x, y, npoints)

    # normalize area
    area = polyarea(x, y)
    sf = np.sqrt(area)
    x = x / sf
    y = y / sf

    xy = np.column_stack((x, y))

    return xy, sf


def polyarea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_silhouette_score(d, s, o):
    n = np.arange(2, 11)
    silhouette_avg = []
    for num_clusters in n:
        # initialise kmeans
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(d)
        cluster_labels = kmeans.labels_

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
    kmeans = KMeans(n_clusters=idx + 2)
    kmeans.fit(d)

    return [idx + 2, kmeans.labels_, kmeans.cluster_centers_]


def remove_island_pixels(img):
    # convert to unint8 for cv2
    img = img.astype(np.uint8)
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # if components is 1 just return
    if nb_components != 1:
        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        # 5 for now
        min_size = 5

        img = np.zeros((output.shape))
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img[output == i + 1] = 1

    return img


def linear_interpolation(x, y, npoints):
    points = np.array([x, y]).T  # a (nbre_points x nbre_dim) array

    # find points that are closer to x=0 use that as start
    newpoints = np.multiply(points[:, 0], points[:, 1])
    newpoints = np.abs(newpoints)
    idx_sort = np.argsort(newpoints)
    # potentially could miss a point if the intial mask is very clustered
    filtered_points = points[idx_sort[:10]]
    # another filter for greatest y and smallest x
    idx_maxy = np.argsort(filtered_points[:, 1])[-2:]
    idx_minx = np.abs(filtered_points[idx_maxy, 0])
    idx = idx_sort[idx_maxy[idx_minx.argmin()]]

    # idx_min = xnew.argmin()
    points = np.concatenate((points[idx:], points[:idx]))

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)

    alpha = np.linspace(distance.min(), int(distance.max()), npoints)
    interpolator = interpolate.interp1d(distance, points, kind="slinear", axis=0)
    interpolated_points = interpolator(alpha)

    out_x = interpolated_points.T[0]
    out_y = interpolated_points.T[1]

    return out_x, out_y


if __name__ == "__main__":
    main()
