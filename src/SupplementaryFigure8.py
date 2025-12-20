import pandas as pd
import matplotlib.pyplot as plt

metrics = pd.read_csv('quality_metrics_umap_coordinates.csv')

cols2average = ['M/SD:CD11c', 'M/SD:CD21', 'M/SD:CD4', 'M/SD:CD8', 'M/SD:Ki67']
metrics['avgMSD'] = metrics[cols2average].mean(axis=1)

plt.figure()
for tlabel in tissues:
    dft = metrics[metrics['tissue_type']==tlabel]
    plt.plot(dft['avgMSD'],dft['FracImgOfCells'],'.',label=tlabel)
plt.xlabel('average M/SD across channels')
plt.ylabel('Fraction of image covered by tissue')
plt.legend()
plt.savefig('SupplementaryFigure8.png')
plt.show()
