# Things to consider when training MorphoFeatures pipeline on your dataset:

### Segmentation quality:
The pipeline is decently robust to small cell segmentation errors. However, we generally encourage proofreading any segmentation to make sure there are no merge errors that attach pieces of more than 25% of the original cell size, since this could highly distort the texture features. We also recommend plotting a distribution of cell sizes and manually inspecting the outliers.

### Choosing feature vector size:
In case of no proxy task available, which could be used to evaluate different sizes, we recommend rather choosing a bigger vector size (e.g. 512).
The feature vector could be then compressed to remove highly correlated features using, for example, feature clustering or dimentionality reduction techniques.

### Choosing augmentations:
A good starting point is using standard augmentations for the given data modality, often reported in papers using deep learning for classification/segmentation of such data. However, we would strongly encourage visualising the resulting augmented data and consulting a person with expertise in this data modality to confirm that used augmentations do not distort the data or generate unrealistic samples.

### Choosing texture patches resolution:
This parameter mainly depends on the size of the GPU used. Generally, the higher the better.

### Choosing texture patches size:
In our experience, bigger patch sizes do not necessarily work better because they might combine different textures. We would recommend going for the size that could fit a Golgi stack or a couple of medium-sized mitochondria.

### Clustering:
We recommend using community detection methods for clustering the resulting morphological representations. This requires setting a resolution parameter which influences the number of resulting clusters. Our algorithm for setting this parameter and further defining subclusters was as follows:
- Choose an initial parameter that would result in a smaller number of clusters that could be easily analysed manually (10-20)
- Visually inspect the clusters and further subcluster (by increasing clustering resolution) if necessary. We suggest further subclustering in the following cases:
    - cells in a cluster occupy clearly distinct areas in the volume (e.g. subclusters 8.1, 8.2 and 8.3 in Figure 3B);
    - cells have distinct visual appearance (e.g. subclusters 14 and 15 in Figure 3B);
    - gene expression (or other available information about the data) shows high heterogeneity in a cluster (subclustering midgut cells, Figure 5);
    - prior knowledge is available, such as cell types or broader cell classes (e.g. splitting the cluster of foregut neurons and foregut epithelial cells, Figure 9).
