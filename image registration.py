#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


# In[16]:


img1 = cv2.imread('Q3_1.jpg')
img2 = cv2.imread('Q3_2.jpg')
img_gry =cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# In[17]:


img2.shape


# In[18]:


Ix10, Iy10 = np.gradient(img1[:,:,0])
Ix11, Iy11 = np.gradient(img1[:,:,1])
Ix12, Iy12 = np.gradient(img1[:,:,2])
Ix20, Iy20 = np.gradient(img2[:,:,0])
Ix21, Iy21 = np.gradient(img2[:,:,1])
Ix22, Iy22 = np.gradient(img2[:,:,2])
Ix1 = np.zeros((1280,960))
Iy1 = np.zeros((1280,960))
Ix2 = np.zeros((1280,960))
Iy2 = np.zeros((1280,960))
for i in range(1280):
    for j in range(960):
        max_magnitude = max(abs(Ix10[i][j]), abs(Ix11[i][j]), abs(Ix12[i][j]))
        Ix1[i][j] = max_magnitude
for i in range(1280):
    for j in range(960):
        max_magnitude = max(abs(Iy10[i][j]), abs(Iy11[i][j]), abs(Iy12[i][j]))
        Iy1[i][j] = max_magnitude
for i in range(1280):
    for j in range(960):
        max_magnitude = max(abs(Ix20[i][j]), abs(Ix21[i][j]), abs(Ix22[i][j]))
        Ix2[i][j] = max_magnitude
for i in range(1280):
    for j in range(960):
        max_magnitude = max(abs(Iy20[i][j]), abs(Iy21[i][j]), abs(Iy22[i][j]))
        Iy2[i][j] = max_magnitude


# In[19]:


Ix1y1 = Ix1 * Iy1
Ix1x1 = Ix1 * Ix1
Iy1y1 = Iy1 * Iy1
Ix2y2 = Ix2 * Iy2
Ix2x2 = Ix2 * Ix2
Iy2y2 = Iy2 * Iy2


# In[20]:


def gaus_conv(matrix):
    kernel = np.array([
        [1,  4,  7,  4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4, 1]
    ], dtype=np.float32)
    kernel /= np.sum(kernel)
    padded_matrix = np.pad(matrix, pad_width=2, mode='constant')
    convolved_matrix = convolve2d(padded_matrix, kernel, mode='valid')
    return convolved_matrix


# In[21]:


S2x_1 = gaus_conv(Ix1x1)
S2y_1 = gaus_conv(Iy1y1)
SxSy_1 = gaus_conv(Ix1y1)
S2x_2 = gaus_conv(Ix2x2)
S2y_2 = gaus_conv(Iy2y2)
SxSy_2 = gaus_conv(Ix2y2)


# In[22]:


plt.imshow(SxSy_2,cmap='gray')


# In[33]:


plt.imshow(S2y_1,cmap='gray')


# In[32]:


plt.imshow(S2x_1,cmap='gray')


# In[24]:


det1 = S2x_1 * S2y_1 - 2* SxSy_1
det2 = S2x_2 * S2y_2 - 2* SxSy_2
trace1 = S2x_1 + S2y_1
trace2 = S2x_2 + S2y_2


# In[25]:


k1 = 0.09
k2 = 0.09
R1 = det1 - k1*trace1*trace1
R2 = det2 - k2*trace2*trace2


# In[26]:


plt.imshow(R1,cmap='gray')


# In[34]:


R1_n = cv2.normalize(R1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
R2_n = cv2.normalize(R2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
cv2.imwrite("res04_score.jpg",R1_n)
cv2.imwrite("res05_score.jpg",R2_n)


# In[28]:


plt.imshow(R2_n,cmap='gray')


# In[61]:


R = R1_n
c,l = R.shape
result = np.zeros((c,l))
rmax = np.max(R) 
for i in range(1,c-1):
    for j in range(1,l-1):
        if R[i,j] > 0.23 * rmax and R[i,j] > np.max(R[i-50:i+50,j+1:j+100]) and R[i,j]>np.max(R[i-50:i+50,j-100:j-1]):
            result[i,j] = 1
x1, y1 = np.where(result == 1)
plt.plot(y1, x1, "r.")
img1 =cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img1)
plt.show()


# In[62]:


R2 = R2_n
c2,l2 = R2.shape
result2 = np.zeros((c2,l2))
rmax2 = np.max(R2) 
for i in range(1,c2-1):
    for j in range(1,l2-1):
        if R2[i,j] > 0.2 * rmax2 and R2[i,j] > np.max(R2[i-50:i+50,j+1:j+100]) and R2[i,j]>np.max(R2[i-50:i+50,j-100:j-1]): 
            result2[i,j] = 1
x2, y2 = np.where(result2 == 1)
plt.plot(y2, x2, "r.")
img2 =cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.show()


# In[63]:


x1.shape


# In[64]:


y2.shape


# In[65]:


n =80
rows1 = 36
feature1 = [[] for i in range(rows1)]
for k1 in range(36):
    for i1 in range(n):
        for j1 in range(n):
            feature1[k1] = img1[x1[k1]-(n//2):x1[k1]+(n//2),y1[k1]-(n//2):y1[k1]+(n//2),:].flatten()
rows2 = 33
feature2 = [[] for i in range(rows2)]
for k2 in range(33):
    for i2 in range(n):
        for j2 in range(n):
            feature2[k2] = img2[x2[k2]-(n//2):x2[k2]+(n//2),y2[k2]-(n//2):y2[k2]+(n//2),:].flatten()


# In[100]:



p1 = np.zeros((36),dtype=int)
p2 = np.zeros((36),dtype=int)
d1 = np.zeros((36))
d2 = np.zeros((36))
for q in range(36):
    compare = np.zeros((33))
    for b in range(33):
        compare[b] = np.linalg.norm(feature1[q]-feature2[b],ord=1)
    p1[q] = int(np.argmin(compare))
    d1[q] = np.min(compare)
    compare[p1[q]] = 100000000
    p2[q] = int(np.argmin(compare))
    d2[q] = np.min(compare)
    if d1[q]/d2[q] > 0.99:
        p1[q] = 0
    


# In[103]:



p1_2 = np.zeros((33),dtype=int)
p2_2 = np.zeros((33),dtype=int)
d1_2 = np.zeros((33))
d2_2 = np.zeros((33))
for q2 in range(33):
    compare2 = np.zeros((36))
    for b2 in range(36):
        compare2[b2] = np.linalg.norm(feature2[q2]-feature1[b2],ord=2)
    p1_2[q2] = int(np.argmin(compare2))
    d1_2[q2] = np.min(compare2)
    compare2[p1_2[q2]] = 100000000
    p2_2[q2] = int(np.argmin(compare2))
    d2_2[q2] = np.min(compare2)
    if d1_2[q2]/d2_2[q2] > 0.99:
        p1_2[q2] = 0
    


# In[217]:


src1 = y1[9],x1[9]
src2 = y1[32],x1[32]
src3 = y1[16],x1[16]
src4 = y1[19],x1[19]
src5 = y1[1],x1[1]


# In[220]:


dst1 = y2[13]+960,x2[13]
dst2 = y2[29]+960,x2[29]
dst3 = y2[21]+960,x2[21]
dst4 = y2[22]+960,x2[22]
dst5 = y2[2]+960,x2[2]


# In[223]:


imge = cv2.hconcat([img1, img2])
cv2.line(imge,src1,dst1,(255, 0, 0), 5)
cv2.line(imge,src2,dst2,(255, 0, 0), 5)
cv2.line(imge,src3,dst3,(255, 0, 0), 5)
cv2.line(imge,src4,dst4,(255, 0, 0), 5)
cv2.line(imge,src5,dst5,(255, 0, 0), 5)
plt.imshow(imge)


# In[156]:


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
good_matches = []
ratio_threshold = 0.2
for m, n in matches:
    if m.distance < ratio_threshold * n.distance:
        good_matches.append(m)
corresponding_points1 = []
corresponding_points2 = []

for match in good_matches:
    point1 = keypoints1[match.queryIdx].pt
    point2 = keypoints2[match.trainIdx].pt
    corresponding_points1.append(point1)
    corresponding_points2.append(point2)


# In[157]:


output_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                               matchColor=(255, 0, 0), singlePointColor=None, flags=2)


# In[158]:


plt.imshow(output_image)


# In[203]:


image1 = cv2.imread('img1.jpg')
image2 = cv2.imread('img2.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 0.2)
inlier_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]
matching_result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matching_result)


# In[ ]:




