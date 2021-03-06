## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m`
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

from matplotlib import use
use('TkAgg')
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import scipy.io

from solutions.ex7.featureNormalize import featureNormalize
from solutions.ex7.pca import pca
from solutions.ex7.projectData import projectData
from solutions.ex7.recoverData import recoverData
from solutions.ex7.kMeansInitCentroids import kMeansInitCentroids
from solutions.ex7.runkMeans import runkMeans
from solutions.ex7.plotDataPoints import plotDataPoints
from solutions.show import show

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize

print('Visualizing example dataset for PCA.')
#  The following command loads the dataset. You should now have the 
#  variable X in your environment
data = scipy.io.loadmat('ex7data1.mat')
X = data['X']

#  Visualize the example dataset
plt.scatter(X[:, 0], X[:, 1], marker='o', color='b', facecolors='none', lw=1.0)
plt.axis([0.5, 6.5, 2, 8])
plt.axis('equal')
show()

input('Program paused. Press Enter to continue...')  

## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('Running PCA on example dataset.')

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S, V = pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
mu2 = mu + 1.5 * S.dot(U.T)
plt.plot([mu[0], mu2[0, 0]], [mu[1], mu2[0, 1]], '-k', lw=2)
plt.plot([mu[0], mu2[1, 0]], [mu[1], mu2[1, 1]], '-k', lw=2)
show()

print('Top eigenvector: ')
print(' U(:,1) = %f %f ', U[0, 0], U[1, 0])
print('(you should expect to see -0.707107 -0.707107)')

input('Program paused. Press Enter to continue...')  


## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('Dimension reduction on example dataset.')

#  Plot the normalized dataset (returned from pca)
plt.figure()
plt.scatter(X_norm[:, 0], X_norm[:, 1], marker='o', color='b', facecolors='none', lw=1.0)
plt.axis([-4, 3, -4, 3]) #axis square
plt.axis('equal')
show()

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: %f', Z[0])
print('(this value should be about 1.481274)')

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: %f %f' % (X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)')

#  Draw lines connecting the projected points to the original points
plt.scatter(X_rec[:, 0], X_rec[:, 1], marker='o', color='r', facecolor='none', lw=1.0)
for i in range(len(X_norm)):
    plt.plot([X_norm[i, 0], X_rec[i, 0]], [X_norm[i, 1], X_rec[i, 1]], '--k')

show()
input('Program paused. Press Enter to continue...')  

## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#

def displayData(X):
    """displays 2D data
      stored in X in a nice grid. It returns the figure handle h and the
      displayed array if requested."""

    # Compute rows, cols
    m, n = X.shape
    example_width = int(round(np.sqrt(n)))
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m / display_rows)

    # Between images padding
    pad = 1
    h = (int(pad + display_rows * (example_height + pad)))
    w = (int(pad + display_cols * (example_width + pad)))
    # Setup blank display
    display_array = - np.ones([h, w])

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in np.arange(display_rows):
        for i in np.arange(display_cols):
            if curr_ex > m:
                break
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, : ]))
            rows = np.array([pad + j * (example_height + pad) + x for x in np.arange(example_height+1)]).astype('int')
            cols = np.array([pad + i * (example_width + pad)  + x for x in np.arange(example_width+1)]).astype('int')
            #             X[curr_ex, :].reshape(example_height, example_width) / max_val
            display_array[min(rows):max(rows), min(cols):max(cols)]
            display_array[min(rows):max(rows), min(cols):max(cols)] = X[curr_ex, :].reshape(example_height, example_width) / max_val
            curr_ex = curr_ex + 1
        if curr_ex > m:
            break

        # Display Image
    display_array = display_array.astype('float32')
    plt.imshow(display_array.T)
    plt.set_cmap('gray')
    # Do not show axis
    plt.axis('off')
#     show()

print('Loading face dataset.')

#  Load Face dataset
data = scipy.io.loadmat('ex7faces.mat')
X = data['X']

#  Display the first 100 faces in the dataset
displayData(X[0:100, :])

input('Program paused. Press Enter to continue...')  

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('Running PCA on face dataset.\n(this might take a minute or two ...)\n\n')

#  Before running PCA, it is important to first normalize X by subtracting 
#  the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S, V = pca(X_norm)

#  Visualize the top 36 eigenvectors found
displayData(U[:, 1:36].T)

input('Program paused. Press Enter to continue...')  

## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('Dimension reduction for face dataset.')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print('%d %d' % Z.shape)

input('Program paused. Press Enter to continue...')  

## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('Visualizing the projected (reduced dimension) faces.')

K = 100
X_rec  = recoverData(Z, U, K)

# Display normalized data
plt.subplot(1, 2, 1)
displayData(X_norm[:100,:])
plt.title('Original faces')
plt.axis('equal')

# Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2)
displayData(X_rec[:100,:])
plt.title('Recovered faces')
plt.axis('equal')
show()
input('Program paused. Press Enter to continue...')  


## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

# Re-load the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
A = scipy.misc.imread('bird_small.png')

# If imread does not work for you, you can try instead
#   load ('bird_small.mat')

A = A / 255.0
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3)
K = 16 
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = np.floor(np.random.random(1000) * len(X)) + 1

#  Setup Color Palette

#  Visualize the data and centroid memberships in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sel = np.array(sel).astype('int')
Xs = np.array([X[s] for s in sel])
xs = Xs[:, 0]
ys = Xs[:, 1]
zs = Xs[:, 2]
cmap = plt.get_cmap("jet")
idxn = sel.astype('float')/max(sel.astype('float'))
colors = cmap(idxn)
# ax = Axes3D(fig)
ax.scatter3D(xs, ys, zs=zs, edgecolors=colors, marker='o', facecolors='none', lw=0.4, s=10)

plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
show()
input('Program paused. Press Enter to continue...')

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
U, S, V = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
plt.figure()
zs = np.array([Z[s] for s in sel])
idxs = np.array([idx[s] for s in sel])

# plt.scatter(zs[:,0], zs[:,1])
plotDataPoints(zs, idxs)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
show()
input('Program paused. Press Enter to continue...')
