import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.transform import estimate_transform
import matplotlib.pyplot as plt
import os


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

def computeCylindrical(im,f):
    x_len = im.shape[1]
    y_len = im.shape[0]
    
    x_cen = x_len/2
    y_cen = y_len/2

    cyl_img = np.zeros(im.shape,im.dtype)
    cyl_coords = np.zeros((im.shape[0],im.shape[1],2),dtype=np.float)
    for curr_x in range(im.shape[1]):
        for curr_y in range(im.shape[0]):
            cyl_coords[curr_y,curr_x,0] = np.arctan((curr_x - x_cen)/f)
            cyl_coords[curr_y,curr_x,1] = 1.*(curr_y - y_cen)/np.sqrt((curr_x - x_cen)**2 + f**2)
            corr_x = np.around(x_cen + f*np.tan(1.*(curr_x-x_cen)/f)).astype(int)
            corr_y = np.around(y_cen + np.sqrt((corr_x-x_cen)**2 + f**2)*( (curr_y - y_cen)/f ) ).astype(int)
            if corr_x<0 or corr_x>=x_len or corr_y<0 or corr_y>=y_len:
                continue
            cyl_img[curr_y,curr_x] = im[corr_y,corr_x]
    # cv2.imshow('warp',cyl_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cyl_img,cyl_coords
            
def ransacA(src, dst, num_iter=2000, tol=8):
    pts1 = np.concatenate((src,np.ones((len(src),1),src.dtype)),axis=-1)
    pts2 = np.concatenate((dst,np.ones((len(src),1),src.dtype)),axis=-1)
    
    max_inliers = 0
    # print(pts2.shape)
    bestA = np.zeros((2,3),dtype=np.float)
    for ind in range(num_iter):
        random_inds = np.random.choice(len(pts1),2)
        p1 = pts1[random_inds,:-1] 
        p2 = pts2[random_inds,:-1]
        trans = np.mean(p2-p1,axis=0)
        curr_A = np.eye(3)[:-1]
        curr_A[:,-1] = trans

        # curr_A = cv2.estimateRigidTransform(p1,p2,fullAffine=False)
        # print(curr_A)
        if len(np.shape(curr_A)) == 0:
            continue
        comp_pts1 = (curr_A@pts1.T).T
        dist = np.linalg.norm(comp_pts1 - pts2[:,:-1],axis=1)
        inliers = np.where(dist<tol)[0]
        # print(len(inliers))
        if len(inliers)>max_inliers:
            max_inliers = len(inliers)
            bestH = curr_A
        # print(match_pts)
        # print(comp_p1[:,:5]/comp_p1[2,:5])
    print(max_inliers)
    return bestH


if __name__ == '__main__':
# Number of images to be stitched together. In init_ind is an index that is basically
# used to identify the starting point of images that have been named sequentially 
    no_images = 16
    init_ind=1
    # im_folder = 'data/cyl_pano/'
    # im1 = cv2.imread(('data/cyl_pano/im'+str(init_ind)+'_c.jpg'))
    # im2 = cv2.imread(('data/cyl_pano/im' + str(init_ind+1) + '_c.jpg'))
    im_folder = 'data/Room360/data3/'
    images = sorted(os.listdir(im_folder))
    print(sorted(images))
    im1 = cv2.imread((im_folder+images[0]))
    im2 = cv2.imread((im_folder+images[1]))
    print('Image 0 path:', im_folder+images[0])
    print('Image 1 path:', im_folder+images[1])
    foc_len = 3500/(1.22)

# Project to cylindrical coordinates and unwrap the cylinder
    # cyl1, _ = computeCylindrical(im1,foc_len)
    # cyl2, _ = computeCylindrical(im2,foc_len)

    cyl1 = im1
    cyl2 = im2
    for image_index in range(len(images)):
        print('Current image number:', image_index)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(cyl1,None)
        kp2, des2 = orb.detectAndCompute(cyl2,None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        src = []
        dst = []

        for match in matches[:40]:
            src.append(np.array(kp2[match.trainIdx].pt))
            dst.append(np.array(kp1[match.queryIdx].pt))
        
        src = np.stack(src,axis=0)
        dst = np.stack(dst,axis=0)
        
        img3 = cv2.drawMatches(cyl1,kp1,cyl2,kp2,matches[:40],None, flags=2)

        cv2.imshow('img3',img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Compute the affine transformation matrix
        # A = cv2.estimateRigidTransform(src,dst,fullAffine=False)
        # A,_ = cv2.findHomography(src,dst,method=cv2.RANSAC)
        A = ransacA(src,dst)
        print(A)
        scale_mat,final = imageStitching_noClip(cyl1,cyl2,A)
        cyl1 = final

        # Load the next image and bring it to the same scale as the image stitched so far
        next_path = im_folder + images[image_index+2]
        print('Current image path:', next_path)
        im2 = cv2.imread(next_path)
        cyl2, _ = computeCylindrical(im2,foc_len)
        scale_fac = scale_mat[0,0]
        # cv2.imshow('img4',im2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cyl2 = cv2.warpAffine(cyl2,scale_mat,(np.floor(cyl2.shape[1]*scale_fac).astype(int),np.floor(cyl2.shape[0]*scale_fac).astype(int)))
        # cv2.imshow('img5',cyl2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        final2 = cv2.resize(final,(1600,400))
        cv2.imshow('final',final2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()