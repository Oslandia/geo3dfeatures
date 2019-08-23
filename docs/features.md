# Geometric features

## Raw features

3D data are first characterized with their relative positioning into the global
point cloud. This positioning is expressed as three basic coordinates into the
Cartesian coordinate system, along `x`-, `y`- and `z`-axis.

See [Wikipedia](https://en.wikipedia.org/wiki/Cartesian_coordinate_system) for
more details on this coordinate system.

By considering a local neighborhood with respect to a point within the point
cloud, one can compute the set covariance matrix over the three dimensions `x`,
`y` and `z`. Some of the following sets of features are defined with the
covariance matrix eigenvalues $`\lambda_i`$, $`\forall i \in {1, 2, 3}`$, and
their normalized version $`e_i`$, where
$`e_i=\frac{\lambda_i}{\sum_{j\in{1, 2, 3}}{\lambda_j}}`$, $`\forall
i \in {1, 2, 3}`$.

As a remark, PCA software outputs correspond to singular values $`s_i`$, with
$`\lambda_i = s_i^2`$ (*ex* `sklearn` in Python).

## Barycentric coordinates

The importance of these features was first highlighted
by [Brodu et al (2011)](#references).

$`\alpha`$ and $`\beta`$ denotes the barycentric coordinates of normalized
eigenvalues. Some theoretical elements on the conversion between Cartesian and
barycentric coordinates are detailed
on
[Wikipedia](https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Conversion_between_barycentric_and_Cartesian_coordinates).

By assuming that `pca` represents the output of a principle component analysis
over the point local neighborhood, one can compute such features by doing:
```python
geo3dfeatures.features.triangle_variance_space(pca)
```

## 3D properties

Given the `nb_neighbors` the number of neighbors, `neighbor_z` the *z*
coordinates of neighboring points, the distances `dist` between the point of
interest and its neighbors and `pca` the resulting PCA of the neighboring set,
one can compute a set of 3D properties that are detailed below.

```python
geo3dfeatures.features.val_range(neighbor_z)
geo3dfeatures.features.std_deviation(neighbor_z)
geo3dfeatures.features.radius_3D(dist)
geo3dfeatures.features.density_3D(radius3D, nb_neighbors)
geo3dfeatures.features.verticality_coefficient(pca)
```

See [Weinmann et al (2015)](#references) for theoretical details.

### z range

$`\Delta_{z,k}`$ is the vertical range of height values within the neighborhood

### z standard deviation

$`\sigma_{z,k}`$ is the standard deviation of height values within the
neighborhood

### radius

$`r_k`$ represents the radius of the spherical neighborhood encapsulating the
$`k`$ closest neighbors

### density

$`D`$ denotes local point density within the neighborhood

### verticality

$`V`$ is derived from the vertical component of the normal vector, provides the
vertical trend of the local neighborhood.

## 3D features

Given the covariance matrix eigenvalues `lambda`, and their normalized version
`e`, one can compute the feature set with:

```python
geo3dfeatures.features.curvature_change(e)
geo3dfeatures.features.linearity(e)
geo3dfeatures.features.planarity(e)
geo3dfeatures.features.omnivariance(e)
geo3dfeatures.features.anisotropy(e)
geo3dfeatures.features.eigenentropy(e)
geo3dfeatures.features.curvature_change(e)
geo3dfeatures.features.val_sum(lambda)
```

It returns a list with the features detailed below.

For more theoretical details, see [Weinmann et al (2015)](#references)
citing [West et al. (2004)](#references)
and [Pauly et al. (2003)](#references).

### curvature_change


$`C_{\lambda} = \frac{\lambda_3}{\sum_i^3 \lamdba_i}`$

équivalent à

$`C_{\lambda} = e_3`$

### linearity

$`L_{\lambda} = \frac{e_1-e_2}{e_1}`$

### planarity

$`P_{\lambda} = \frac{e_2-e_3}{e_1}`$

### scattering

$`S_{\lambda} = \frac{e_3}{e_1}`$

### omnivariance

$`O_{\lambda} = \sqrt[3]{e_1*e_2*e_3}`$

### anisotropy

$`C_{\lambda} = \frac{e_1-e_3}{e_1}`$

### eigenentropy

$`E_{\lambda} = - \sum_{i=1}^{3}{e_i*\ln{e_i}}`$

### sum of eigenvalues

$`\Sigma_{\lambda} = \lambda_1 + \lambda_2 + \lambda_3`$

## 2D properties

If focusing on man-made structure, there may be some symmetric and orthogonal
patterns into the point cloud. In this way, considering the point cloud 2D
projection may highlight new elements. Radius and density are computed on the
same manner than for the 3D point cloud.

See [Weinmann et al (2015)](#references) for further explanations.

Considering `point` the 2D coordinates of the point of interest, `neighbors`
the 2D coordinates of neighboring points, and `nb_neighbors` the number of
neighbors, one gets the 2D properties with following functions:

```python
geo3dfeature.features.radius_2D(point, neighbors)
geo3dfeature.features.density_2D(radius2D, nb_neighbors)
```

### radius

$`r_{k, 2D}`$ represents the radius of the spherical neighborhood encapsulating
the $`k`$ closest neighbors in the 2D space.

### density

$`D_{2D}`$ denotes local point density within the neighborhood in the 2D space.

## 2D features

[Weinmann et al (2015)](#references) propose two additional 2D features built
on the model of 3D features: the sum and the ratio of eigenvalues computed over
2D data.

Given the covariance matrix eigenvalues `lambdas`, computed on 2D data
projection, one can compute the feature set with:

```python
geo3dfeatures.features.val_sum(lambdas)
geo3dfeatures.features.eigenvalue_ratio_2D(lambdas)
```

### sum of eigenvalues

$`\Sigma_{\lambda, 2D} = \lambda_1 + \lambda_2`$

### ratio of eigenvalues

$`R_{\lambda, 2D} = \frac{e_2}{e_1}`$

## Accumulation features

The accumulation features are 2D-based features, as they are based on an
alternative neighborhood definition. Instead of considering the $`k`$ nearest
neighborhood (`kNN`), one has to design bins of fixed size, and to sort each
point into its corresponding bin regarding the 2D projection of the point
cloud.

This neighborhood definition has been used in
and [Monnier et al. (2012)](#references) [Weinmann et al (2015)](#references),
for instance.

In order to compute this set of features, one has to assign each point to its
corresponding bin with:

```python
geo3dfeatures.features.accumulation_2d_neighborhood(point_cloud, bin_size, buf)
```

where `point_cloud` represents the point 3D coordinates into the point cloud,
`bin` the bin size (expressed in the same unity than those of the point cloud)
and `buf` a buffer that helps managing the extrem points in the point cloud. As
this method returns a DataFrame with accumulated density, z-range and
z-standard-deviation associated to each point in the cloud, accessing the
feature value is straightforward.

### accumulation density

$`M`$ is the number of points that lies into the point bin.

### accumulated z range

$`\Delta_{z}`$ represents the maximum height difference within the point bin.

### accumulated z standard deviation

$`\sigma_{z}`$ is the standard deviation of height values within the point bin.

# Color features

In order to manage color differences and improve automatic point classification
algorithms we store red, green and blue pixel components.

# References

- Nicolas Brodu, Dimitri Lague, 2011. [3D Terrestrial lidar data classification of complex natural scenes using a multi-scale dimensionality criterion: applications in geomorphology](https://arxiv.org/abs/1107.0550). arXiv:1107.0550.

- Fabrice Monnier, Bruno Vallet, Bahman Soheilian, 2012. [Trees detection from laser point clouds acquired in dense urban areas by a mobile mapping system](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/I-3/245/2012/isprsannals-I-3-245-2012.pdf). ISPRS Annals of the Phogrammetry, Remote Sensing and Spatial Information Sciences. 1-3, pp.245-250.

- Mark Pauly, Richard Keiser, Markus Gross, 2003. [Multi‐scale Feature Extraction on Point‐Sampled Surfaces](https://graphics.stanford.edu/courses/cs468-03-fall/Papers/Pauly_FeatureExtraction.pdf). Computer Graphics Forum, 22(3):281-289.

- Martin Weinmann, Boris Jutzi, Stefan Hinz, Clément Mallet, 2015. [Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers](http://recherche.ign.fr/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf). ISPRS Journal of Photogrammetry and Remote Sensing, vol 105, pp 286-304.

- Karen F. West, Brian N. Webb, James R. Lersch, Steven Pothier, 2004. [Context-driven automated target detection in 3D Data](https://www.researchgate.net/publication/241586585_Context-driven_automated_target_detection_in_3D_data). Proceedings of SPIE - The International Society for Optical Engineering 5426, September 2004.
