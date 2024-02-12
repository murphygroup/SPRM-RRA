import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import defaultdict
from pathlib import Path
import os

# Replace with the paths to your images
# root = Path('/Users/tedzhang/Desktop/sprm_outputs_3')
root = Path('/Users/tedzhang/Desktop/CMU/hubmap/SPRM/Cellar/figure_9/')
# image_paths = ['/Users/tedzhang/Desktop/CMU/hubmap/SPRM/sprm_outputs_manu/reg1_stitched_expression.ome.tiff-clusterbyTotalpercell.png', '/Users/tedzhang/Desktop/CMU/hubmap/SPRM/sprm_outputs_manu/reg1_stitched_expression.ome.tiff-clusterbyMeanspercell.png', '/Users/tedzhang/Desktop/CMU/hubmap/SPRM/sprm_outputs_manu/reg1_stitched_expression.ome.tiff-clusterbyMeansAll.png',
#                '/Users/tedzhang/Desktop/CMU/hubmap/SPRM/sprm_outputs_manu/reg1_stitched_expression.ome.tiff-clusterbyCovarpercell.png', '/Users/tedzhang/Desktop/CMU/hubmap/SPRM/sprm_outputs_manu/reg1_stitched_expression.ome.tiff-Cluster_Shape.png', '/Users/tedzhang/Desktop/CMU/hubmap/SPRM/sprm_outputs_manu/reg1_stitched_expression.ome.tiff-clusterbytSNEAllFeatures.png',
#                '/Users/tedzhang/Desktop/CMU/hubmap/SPRM/sprm_outputs_manu/reg1_stitched_expression.ome.tiff-Cluster_ShapeNormalized.png', '/Users/tedzhang/Desktop/CMU/hubmap/SPRM/sprm_outputs_manu/reg1_stitched_expression.ome.tiff-clusterbyUMAP.png']


# image_names = ['clusterbyTotalpercell.png', 'clusterbyMeanspercell.png', 'clusterbyMeansAll.png',
#                'clusterbyCovarpercell.png', 'Cluster_Shape.png', 'Cluster_ShapeNormalized.png', 'clusterbytSNEAllFeatures.png',
#                'clusterbyUMAP.png']

# image_paths = []
# #
# for i in image_names:
#     image_paths.append(list(root.glob('r*-'+i))[0])

all_files = os.listdir(root)
only_files = [os.path.join(root, f) for f in all_files if os.path.isfile(os.path.join(root, f))]
# only_files.sort() #for figure 8 only
# images = [Image.open(img_path) for img_path in image_paths]
images = [Image.open(img_path) for img_path in only_files]

### swap ###
# figure 8
# a = images[1]
# images[1] = images[4]
# images[4] = a

#figure 9
a = images.pop(1)
images.append(a)
###########

#recolor
# pixels = images[4].load()
# for y in range(images[4].size[1]):  # For each row
#     for x in range(images[4].size[0]):  # For each column
#         r, g, b, a = pixels[x, y]
#
#         if (r, g, b) == (55, 126, 184):  # If the pixel is blue
#             pixels[x, y] = (77, 175, 74, a)  # Change it to green
#         if (r, g, b) == (77, 175, 74):  # If the pixel is green
#             pixels[x, y] = (228, 26, 28, a)  # Change it to red
#         if (r, g, b) == (228, 26, 28):  # If the pixel is red
#             pixels[x, y] = (77, 175, 74, a)  # Change it to green

# plt.figure(figsize=(20, 20))

# fig, axs = plt.subplots(3, 3, figsize=(10, 10))
# fig, axs = plt.subplots(4, 2, figsize=(30, 30))
fig, axs = plt.subplots(3, 2, figsize=(50, 50))

# names = ['Cell Types', 'UMAP Subtypes', 'Shape Normalized Subtypes', 'Covariance Subtypes', 'Mean-all Subtypes', 'Total Subtypes','Shape Unnormalized Subtypes',
#          'Mean-Cell Subtypes', 'tSNE Subtypes']
names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
# names = ['A', 'B', 'C', 'D', 'E', 'F']

#1 - figure 9
#3 - figure 8

# axs[0, 0].imshow(images[1])
# axs[0, 0].axis('off')
# axs[0, 0].set_title(names[0])
# #
# names.pop(0)
# images.pop(0)


count = 0
for i in range(3):
    for j in range(2):
# for i in range(3):
# #     for j in range(3):
#         if i == 0 and j == 0:
#             continue
        # axs[i, j].subplot(4,2,i+1)
        axs[i, j].imshow(images[count])
        axs[i, j].axis('off')# Optionally turn off the axis.
        axs[i, j].set_title(names[count], fontweight='bold', fontsize=50)

        # axs[i, j].imshow(images[count])
        # axs[i, j].axis('off')
        # axs[i, j].set_title(names[count])
        count += 1

plt.subplots_adjust(wspace=1, hspace=-0.4)
# plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.tight_layout()
plt.savefig('/Users/tedzhang/Desktop/CMU/hubmap/SPRM/MANUSCRIPT/output/figure_9_panels')
# plt.show()
