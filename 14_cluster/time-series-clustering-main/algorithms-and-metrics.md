# **Time series clustering algorithms**

**Time Series Clustering** is an unsupervised technique. 

The objective is to maximize data similarity within clusters and minimize it across clusters.

## 1. Hierarchical clustering

Hierarchical clustering produces a nested clusters of series whih can be represented graphically. 

Example: BIRCH ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html))

*   Agglomerative
  - Each observation is initially a single cluster, then forms a cluster with closest observation
*   Divisive
  - All observations are treated as a single cluster, then repeatedly each observation separated from the most farest one until each observations create a single cluster


In order to decide how the merging is performed, a (dis)similarity measure between groups should be specified, in addition to the one that is used to calculate pairwise similarities.

Specific number of clusters does not need to be specified. In order to get the desired number of clusters a threshhold can be used.

**Drawbacks:**


1.   Small change to data requries restarting the whole process
2.   High computational complexity O(n^2)

---------------------------------------------------

## 2. Partitional clustering

Example: K-means, fuzzy c-means (FCM), Self organising map (SOM), CLARANS, K-medoids

Data is divided in predifined number of clusters. Applies distance or correlation functions to determine similarity between data (Euclidian, Manhattan distances, Pearson correlation coefficient)

First, k centroids are randomly initialized, usually by choosing k objects from the dataset at random; these are assigned to individual clusters. The distance between all objects in the data and all centroids is calculated, and each object is assigned to the cluster of its closest centroid. Then each centroid is updated, so that it is in the center of its cluster. Then, distances and centroids are updated iteratively until a certain number of iterations have elapsed, or no object changes clusters any more.

Fast, applicable for large datasets.

**Drawbacks:**


1.   Each run can provide a different result
2.   Initial number of cluseters is required, bad for time-series.

### *Interesting partitioning clustering algorithms*:

#### 2.1 Self organising map (SOM)

[wiki](https://en.wikipedia.org/wiki/Self-organizing_map)

The SOM is a class of neural network algorithms in the unsupervised learning category. SOM produces a low-dimensional (typically two-dimensional) representation of a higher dimensional data set while preserving the topological structure of the data. 

#### 2.2 Spectral clustering

[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)

#### 2.3 Entropy weigtning clustering

The algorithm is based on the k-means approach to clustering. An initial set of k means are identified as the starting centroids. Observartions are clustered to the nearest centroid according to a distance measure. This defines the initial clustrering. New centroids are then identified based on these clusters.

Weights are then calculated for each variable within each cluster, based on the current clustering. The weights are a measure of the relative importance of each variable with regard to the membership of the observations to that cluster. These weights are then incorporated into the distance function, typically reducing the distance for the more important variables.

[r package](https://rdrr.io/cran/wskm/man/plot.ewkm.html)

#### 2.4 TICC clustering (2017) MULTIVARIATE ONLY

Toeplitz Inverse Covariance-based Clustering (TICC) - a method of clustering multivariate time series subsequences.
The TICC algorithm uses Markov random field structures (MRFs) and simultaneously segments and clusters data in a dynamic way. Unlike K-means and dynamic time warping, TICC looks for structural similarities in the data.

[reference](https://www.ijcai.org/proceedings/2018/0732.pdf) 

[real usage](https://library.seg.org/doi/full/10.1190/geo2021-0478.1)

[python implementation](https://github.com/davidhallac/TICC)

#### 2.5 K-shape clustering (2015)

Uni and multivariate time series. Similar to k-means, but better. Uses shape-based distance.

[python implementation](https://github.com/TheDatumOrg/kshape-python)

---------------------------------------------------

## 3. Density-based clustering

[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)


Example: DBSCAN, HDBSCAN, OPTICS

It is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).

[more about dbscan](https://en.wikipedia.org/wiki/DBSCAN)

[HDBSCAN implementation](https://pypi.org/project/hdbscan/)

Does not require predefined number of clusters.

---------------------------------------------------

## 4. Model-based clustering

[sklearn](https://scikit-learn.org/stable/modules/mixture.html)

The model is based on 3 general assumptions:

1.   We know the number of clusters before we start
2.   Each observation in the data as a certain probability of belonging to each cluster.
3.   The observations within each cluster follow some distribution (with the appropriate dimension): for GMM - Guassian Mixture Model, each cluster follow a normal distribution.

Exmaple: GMM

For clusters assignment Expectation-Maximisation (EM) Algorithm is used: The algorithm works by repeated performing an E-step, which assigns each observation to it’s most likely cluster, and an M-step, which then updates the cluster means and variances based on the assigned observations.

---------------------------------------------------

## 5. Grid-based clustering

In Grid-Based Methods, the space of instance is divided into a grid structure. Clustering techniques are then applied using the Cells of the grid, instead of individual data points, as the base units. The biggest advantage of this method is to improve the processing time.



Partition data space into finite number of cells to form a grid structure and finds clusters (dense regions) from cells.

Example: STING - Statistic Information Grid Approach, CLIQUE

[CLIQUE implementation](https://github.com/georgekatona/Clique)


# **Approaches to apply algorithms**


## 1. Apply to the raw time series

## 2. Apply to preprocessed time series

What could be done to time series:
1.   standardization
2.   smoothing
3.   interpolation
4.   stationarization

## 3. Apply to features of time series, instead of using timeseries themselves

[article](https://robjhyndman.com/papers/wang.pdf)

Possible features:
1. trend
2. seasonality
3. serial correlation
4. non-linear autoregressive structure
5. skewness
6. kurtosis (heavy-tailed distributions)
7. self-similarity (long-range dependence)
8. measures of chaotic dynamics
9. periodicity


# **Similarity measures**

## 1. Euclidian distance

Usual Euclidian distance.

## 2. Dynamic Time Warping (DTW)

[tslearn](https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html)

**Main idea**: Basically DTW is calculated as the squared root of the sum of squared distances between each element in X and its nearest point in Y. Series can be different in time, length and speed(?).


DTW between 𝑥 and 𝑦 is formulated as the following optimization problem:
![DTW](https://miro.medium.com/max/1400/1*mWtMdpkLyHQ4ptNobTK1UA.webp)


**Why length could be different?**

*   DWT uses one-to-many match instead of one-to-one euclidean match

## 3. Shape-based distance

Distance based on coefficient-normalized cross-correlation between two time series

[ts learn](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KShape.html)


# **Time series clustering quality metrics**

## 1. Silhouette Coefficient

[tslearn](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.silhouette_score.html)

The Silhouette Coefficient is defined for each sample and is composed of two scores:


*   **a**: The mean distance between a sample and all other points in the same class.
*   **b**: The mean distance between a sample and all other points in the next nearest cluster.

The Silhouette Coefficient s for a single sample is then given as:
```math
  s = \frac{b - a}{max(a, b)}
```

The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters. The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.


## 2. Dunn index

The Dunn index describes the minimum closest distance between any two clusters divided by the maximum distance between the two farthest points in the cluster. The more Dunn index, the better. 

![Dunn index](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTxHa_pEPYIimqCajk3o4bPFH4vnlSrNRwFpu5EnyCdb60Ii3MyI01okWNMldVF7I4bfQ&usqp=CAU)

[implementation](https://python.engineering/dunn-index-and-db-index-cluster-validity-indices-set/)

## 3. Calinski-Harabasz score 

For a set of data E of size which has been clustered into k clusters, the Calinski-Harabasz score s is defined as the ratio of the between-clusters dispersion mean and the within-cluster dispersion:

```math
s = \frac{\mathrm{tr}(B_k)}{\mathrm{tr}(W_k)} \times \frac{n_E - k}{k - 1}
```

[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html)

## 4. Davies-Bouldin Index

The index is defined as the average similarity between each cluster and its most similar one.

similarity is defined as:

```math
R_{ij} = \frac{s_i + s_j}{d_{ij}}
```

where 

*   s_i - the average distance between each point of cluster i and the centroid of that cluster – also know as cluster diameter.
*   d_i_j - the distance between cluster centroids i and j.

Then the Davies-Bouldin index is defined as:
```math
DB = \frac{1}{k} \sum_{i=1}^k \max_{i \neq j} R_{ij}
```

[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html)