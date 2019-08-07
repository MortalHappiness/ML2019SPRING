# PCA
# type "python pca.py {images_folder} {input_image} {reconstruct_image}" to execute

import os
import sys
import time

import numpy as np 
from skimage.io import imread, imsave

# =====================================

def process(M):
    ans = M - np.min(M)
    ans = ans/np.max(ans)
    ans = (ans * 255).astype(np.uint8)
    return ans

#======================================

_, img_folder, input_image, recon_image = sys.argv

image_shape = (600, 600, 3)

# number of principal components used
nb_pc = 5

t = time.perf_counter()
# =====================================

# load images
img_data = []
for i in range(415):
    img = imread(os.path.join(img_folder, str(i) + '.jpg'))
    img_data.append(img.ravel())

img_data = np.array(img_data).astype('float32')

# Calculate mean & Normalize
mean = np.mean(img_data, axis = 0)
img_data -= mean

# Use SVD to find the eigenvectors
print('Solving svd...')
t0 = time.perf_counter()
u, s, v = np.linalg.svd(img_data.T, full_matrices = False)
t1 = time.perf_counter()
print('Solving svd takes %.3f seconds.' % (t1 - t0))

# =====================================

def reconstruct(img_file):
    # Load image & Normalize
    img = imread(os.path.join(img_folder, img_file))  
    X = img.flatten().astype('float32')
    X -= mean
    
    # Compression
    weight = X.dot(u)
    
    # Reconstruction
    recon = process(weight[:nb_pc].dot(u.T[:nb_pc]) + mean)

    return recon

# ======================================

# reconstruct image
recon = reconstruct(input_image)
imsave(recon_image, recon.reshape(image_shape))

t = time.perf_counter() - t
print('Executing time: %.3f seconds.' % t)

# ======================================

############################################
## The following are for report probelms. ##
############################################

# # report problem 1.a
# average = process(mean)
# imsave('./pca_report/average.jpg', average.reshape(image_shape))

# # report problem 1.b
# for x in range(5):
#   eigenface = process(u[:, x])
#   imsave('./pca_report/%d_eigenface.jpg' % x, eigenface.reshape(image_shape))

# # report problem 1.c
# test_images = [5, 13, 15, 60, 70]
# for i in range(len(test_images)):
#   num = test_images[i]
#   recon = reconstruct('%d.jpg' % num)
#   imsave('./pca_report/%d_recon.jpg' % num, recon.reshape(image_shape))

# # report problem 1.d
# for i in range(5):
#   number = s[i] * 100 / sum(s)
#   print(number)