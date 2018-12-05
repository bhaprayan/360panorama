import pickle
import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import cv2

def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...

    if H2to1.shape[0] == 2:
        H2to1 = np.concatenate((H2to1,np.array([0,0,1],ndmin=2)),axis=0)
    # Specify the width that the final image must have
    given_width = 1698

    # Find the extreme points in image 2 (corners), and transform them to the
    # image 1 reference frame 
    extreme_pts2 = np.array([[0,0,1],[0,im2.shape[0],1],[im2.shape[1],0,1],[im2.shape[1],im2.shape[0],1]]).T 
    tf_extr = H2to1@extreme_pts2
    tf_extr = tf_extr[:-1,:]/tf_extr[-1,:]
    # Find the extreme points in image 1 (corners)
    extr_pts = np.concatenate((np.array([[0,0],[0,im1.shape[0]],[im1.shape[1],0],[im1.shape[1],im1.shape[0]]]).T,tf_extr),axis=1)
    
    # Compute the required width and height of the final image with no scaling
    req_width = extr_pts[0,:].max() - extr_pts[0,:].min()
    req_height = extr_pts[1,:].max() - extr_pts[1,:].min()
    req_AR = req_height/req_width
    given_width = np.floor(req_width).astype(int)
    # Compute the dimensions of the output image using the specified width, and
    # the aspect ratio computed above 
    img_dims = (given_width,np.ceil(req_AR*given_width).astype(int))

    # Compute the translation required to ensure that all the extreme points,
    # and thus, both images lie within the final image 
    t_x = -extr_pts[0,:].min()
    t_y = -extr_pts[1,:].min()

    # Construct the scaling and Euclidean (translation only in this case)
    # transformations, and the final M matrix by multiplying the two
    scale_mat = np.diag((given_width/req_width,given_width/req_width,1))
    trans_mat = np.array([[1,0,t_x],[0,1,t_y],[0,0,1]])
    
    # scale_mat = np.diag((given_width/req_width,given_width/req_width))
    # trans_mat = np.array([[1,0,t_x],[0,1,t_y]])
    M_mat = scale_mat@trans_mat


    # Construct, normalize and transform the masks to the final image coordinate
    # system  
    mask1 = np.pad(np.ones((im1.shape[0]-2,im1.shape[1]-2)),(1,1),'constant',constant_values=(0))
    mask2 = np.pad(np.ones((im2.shape[0]-2,im2.shape[1]-2)),(1,1),'constant',constant_values=(0))
    
    mask1 = distance_transform_edt(mask1)
    mask2 = distance_transform_edt(mask2)

    mask1 = cv2.warpAffine(mask1,M_mat[:-1],img_dims)[:,:,np.newaxis]
    mask2 = cv2.warpAffine(mask2,(M_mat@H2to1)[:-1],img_dims)[:,:,np.newaxis]

    mask1 = cv2.warpPerspective(mask1,M_mat,img_dims)[:,:,np.newaxis]
    mask2 = cv2.warpPerspective(mask2,(M_mat@H2to1),img_dims)[:,:,np.newaxis]

    mask1 = mask1/mask1.max()
    mask2 = mask2/mask2.max()
    
    # Compute warped images 1 and 2, with image 1 only being warped with the M
    # matrix while image 2 is warped using M*H2to1. Use masks for blending the
    # two into the final panorama 
    warped_im1 = cv2.warpPerspective(im1,M_mat,img_dims)
    warped_im2 = cv2.warpPerspective(im2,(H2to1),img_dims)
    pano_im = np.where((mask1+mask2)==0,0,(mask1/(mask1+mask2))*warped_im1 + (mask2/(mask1+mask2))*warped_im2).astype(im1.dtype)
    
    return scale_mat[:-1],pano_im

f = open('fps3_stitched_images.pkl','rb')
stitched_images = pickle.load(f)


# im1 = imresize(stitched_images[0][0],0.5)
# im2 = imresize(stitched_images[1][0],0.5)
# A = stitched_images[0][1]

# A[:,-1] = 0.5*A[:,-1]

# scale_mat, blended = imageStitching_noClip(im1, im2, A)

# im3 = imresize(stitched_images[2][0],0.5)
# B = stitched_images[1][1]

# B[:,-1] = 0.5*B[:,-1]

# B[:,-1] += A[:,-1]

# scale_mat, blended = imageStitching_noClip(blended, im3, B)

# plt.imshow(blended);plt.show()

acc_A = stitched_images[0][1]

im1 = stitched_images[20][0]
im2 = stitched_images[21][0]

scale_mat, blended = imageStitching_noClip(im1, im2, acc_A)

plt.imshow(blended)
plt.show()

for pair in range(22,len(stitched_images)):
    acc_A[:,-1] += stitched_images[pair-1][1][:,-1]
    print(acc_A)
    scale_mat, blended = imageStitching_noClip(blended, stitched_images[pair][0], acc_A)
    print('Stitched pair:',pair)
    plt.imshow(blended)
    plt.show()

plt.imshow(blended)
plt.show()