from skimage.util.shape import view_as_windows
from scipy.sparse import coo_matrix
import cv2
import numpy as np

def windows_from_image(img, window_size, stride):
    color_dim = (img.shape[2],) if img.ndim == 3 else ()
    windows = view_as_windows(np.squeeze(img), (window_size,window_size) + color_dim,stride)
    return windows.reshape((-1, window_size, window_size) + color_dim), windows.shape[0]

def plot_windows(windows):
    fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(10,10))
    for i in range(rows):
        for j in range(cols):
            axs[i,j].imshow(windows[i][j],cmap='gray', vmin=0, vmax=255)

def image_from_windows(windows,rows, block_size, stride,original_shape):
    offsets = [(np.repeat(np.arange(i*stride,i*stride+block_size),block_size).reshape((block_size,block_size)),
              np.repeat(np.arange(j*stride,j*stride+block_size),block_size).reshape((block_size,block_size)).T) for i in range(rows) for j in range(rows) ]

    as_image = np.zeros(original_shape) 
    images = [coo_matrix((np.ravel(windows[i]), (np.ravel(a), np.ravel(b))), shape=original_shape) for i,(a,b) in enumerate(offsets)]
    ones   = [coo_matrix((np.ravel(np.ones((block_size,block_size))), (np.ravel(a), np.ravel(b))), shape=original_shape) for i,(a,b) in enumerate(offsets)]
    return (np.sum(images) / np.sum(ones))

def pred_to_uint8(pred):
    return (255*pred).astype('uint8')
    
def predict_from_image(model, img, window_size=256, stride=32):
    win,rows = windows_from_image(img,window_size,stride)
    predicted = model.predict(win)
    back_pred = image_from_windows(predicted,rows,window_size,stride,(img.shape[0],img.shape[0]))
    return pred_to_uint8(back_pred)

def pad_border(img, adj_border_size):
    left_bot_b = int(adj_border_size // 2)
    right_top_b = int(adj_border_size-left_bot_b)
    return cv2.copyMakeBorder(img, right_top_b, left_bot_b, left_bot_b, right_top_b, cv2.BORDER_CONSTANT, value=[0,0,0])

def rotate_image(mat, angle):
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def predict_from_image_rotated(model, img, window_size=256, stride=32, rotations=[*(np.arange(8) * 45)], pad=True):
    num_rot = len(rotations)
    pre_agg_imgs = np.zeros((num_rot,img.shape[0],img.shape[1]))
    border_size = window_size/2
    
    for rot in range(num_rot):
        
        alpha, inv_alpha = rotations[rot], 360-rotations[rot]
    
        # rotate image
        rot_img = rotate_image(img, alpha) 
        
        if pad==False and all(i % 90 == 0 for i in rotations): 
            pad_rot_img = rot_img
        else: 
            # pad image in order to not have less windows on edges & keep stride multiple
            adj_border_size = (-(-(rot_img.shape[1]+border_size*2) // stride) * stride - rot_img.shape[1]) 
            pad_rot_img = pad_border(rot_img, adj_border_size)

        # retrieve windows from padded / rotated image 
        win,rows = windows_from_image(pad_rot_img, window_size, stride)

        # predict the windows
        predicted = model.predict(win)

        # reconstruct image
        back_pred = image_from_windows(predicted,rows,window_size,stride,(pad_rot_img.shape[0],pad_rot_img.shape[0]))

        # unrotate and remove padding
        unrotated_img = rotate_image(back_pred, inv_alpha)
        img_ctr = unrotated_img.shape[1] // 2
        dx = dy = img.shape[1]/2
        retrieved_img = unrotated_img[int(img_ctr-dx):int(img_ctr+dx) , int(img_ctr-dy):int(img_ctr+dy)]
        
        pre_agg_imgs[rot] = retrieved_img
    
    pred = np.mean(pre_agg_imgs, axis=0)
    return pred_to_uint8(pred)
