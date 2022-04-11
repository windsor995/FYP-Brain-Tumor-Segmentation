import os
from re import S, X
import zipfile
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import nibabel as nib
import SimpleITK as sitk
import math
import cc3d

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from dltk.core.losses import dice_loss
from dltk.io.augmentation import *

from scipy import ndimage

from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from scipy.spatial.distance import cdist

import cv2 as cv
import random



def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    if np.all(volume==0):
        return volume
    
    for ii in range(volume.shape[0]):
        temp = volume[ii,:,:].copy()
        if np.all(temp==0):
            continue
        std = np.std(temp[temp>0], dtype = np.float64)
        average = np.mean(temp[temp>0], dtype = np.float64)
        
        temp[temp>0] = np.divide(np.subtract(temp[temp>0], average), std+1e-8)
        #temp = np.divide(np.subtract(temp, average), std+1e-8)

        volume[ii,:,:] = temp
    
    volume = volume.astype(np.float64) 
    
    return volume  

def percentile_clipping(img):
    for ii in range(img.shape[0]):
        temp = img[ii,:,:]
        if np.all(temp==0):
            continue
        up, low = np.percentile(temp[temp>0], 99), np.percentile(temp[temp>0], 1)
        temp[np.logical_and(temp>up, temp>0)] = up
        temp[np.logical_and(temp<low,temp>0)] = low
        img[ii,:,:] = temp
    return img

def classiffication(volume):
    volume[volume==1]=1
    volume[volume==2]=1
    volume[volume==4]=1
    return volume

def process_both(xpath, xskip, yskip, skip_rate, cropping, multiview):
    xvolume = []
    for ii in xpath:
        xvolume.append(read_nifti_file(ii))
    xvolume = np.array(xvolume)

    if multiview==1:
        xvolume = np.swapaxes(xvolume,1,-1)
        xvolume = np.swapaxes(xvolume,1,2)
    
    elif multiview==2:
        xvolume = np.swapaxes(xvolume,2,-1)

    for ii in range(xvolume.shape[-1]):     
                        
        xone_slice = xvolume[:,:,:,ii]
        """
        if ii==120:
            plt.imshow(xone_slice[0,:,:])
            plt.show()
        """
        #xone_slice = percentile_clipping(xone_slice)
        xone_slice = normalize(xone_slice)

        yield xone_slice

def BRATS_dice_coef(y_true, y_pred, smooth = 1e-7): 
    total_class = 2
    except_last = [1,2]
    
    y_true_f = tf.cast(y_true, tf.float64)
    y_pred_f = tf.cast(y_pred, tf.float64)
    
    y_true_f = tf.cast(K.batch_flatten(K.one_hot(K.cast(tf.squeeze(y_true_f, axis = [-1]), 'int64'), num_classes=total_class)[..., 1:]), dtype = tf.float64)
    y_pred_f = K.batch_flatten(y_pred_f[..., 1:])
    
    intersect = K.sum(y_true_f * y_pred_f, axis = -1)
    denom = K.sum(y_true_f + y_pred_f, axis = -1)
    return K.mean((2.*intersect/(denom+smooth)))

def display(display_list):

    plt.figure(figsize=(15, 7))

    title = ['flair', 't1', 't2', 't1ce', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        
        if len(display_list[i].shape) == 2:
            temp = display_list[i]
            display_list[i] = temp[..., np.newaxis]
        if len(display_list[i].shape) == 4:
            display_list[i] = np.squeeze(display_list[i], axis=0)

        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    #pred_mask = pred_mask[..., 1]
    #return pred_mask
    
    temp = np.zeros([240,240,155])
    threshold = 0.99
    temp[pred_mask[...,1]>=threshold] = 1
    temp[pred_mask[...,1]<threshold] = 0

    return temp
    
    """
    print(pred_mask.max(), np.unravel_index(np.argmax(pred_mask), pred_mask.shape))
    np.set_printoptions(threshold=np.inf)
    print(pred_mask[110:140, 165:200, 89])
    plt.imshow(pred_mask[110:140, 165:200, 89])
    plt.show()
    """
    
    #pred_mask = np.argmax(pred_mask, axis=-1)
    #pred_mask = pred_mask[..., np.newaxis]
    #return pred_mask

def combine_crop(img, dimension, temp=False):
    window_size = 128
    step_size = 32
    #print(img.shape)
    combining = np.zeros(shape=dimension)

    for ii in range(math.ceil((dimension[1]-window_size)/step_size)+1):
        for jj in range(math.ceil((dimension[0]-window_size)/step_size)+1):
            start_1 = jj*step_size
            end_1 = window_size+jj*step_size

            if end_1>dimension[0]:
                start_1 = dimension[0]-window_size
                end_1 = dimension[0]

            start_2 = ii*step_size
            end_2 = window_size+ii*step_size

            if end_2>dimension[1]:
                start_2 = dimension[1]-window_size
                end_2 = dimension[1]

            if temp:
                print(start_1, end_1, start_2, end_2, ii, jj)
                display([create_mask(img[jj+(ii*(math.ceil((dimension[1]-window_size)/step_size)+1)),...])])

            #print(jj+(ii*(math.ceil((dimension[1]-window_size)/step_size)+1)))
            combining[start_1:end_1, start_2:end_2, ...] = combining[start_1:end_1, start_2:end_2, ...] + img[jj+(ii*(math.ceil((dimension[0]-window_size)/step_size)+1)), ...]
            if ii==0 and jj!=0:
                if jj==(math.ceil((dimension[0]-window_size)/step_size)+1-1):
                    combining[start_1:(dimension[0]-(dimension[0]-window_size)%step_size), start_2:end_2, ...] /= 2
                else:
                    combining[start_1:(start_1+window_size-step_size), start_2:end_2, ...] /= 2
            elif ii!=0 and jj!=0:
                if jj==(math.ceil((dimension[0]-window_size)/step_size)+1-1):
                    combining[start_1:(dimension[0]-(dimension[0]-window_size)%step_size), start_2:end_2, ...] /= 2
                    if ii==(math.ceil((dimension[1]-window_size)/step_size)+1-1):
                        combining[(dimension[0]-(dimension[0]-window_size)%step_size):end_1, start_2:(dimension[1]-(dimension[1]-window_size)%step_size), ...] /= 2
                    else:
                        combining[(dimension[0]-(dimension[0]-window_size)%step_size):end_1, start_2:(start_2+window_size-step_size), ...] /= 2
                else:
                    combining[start_1:(start_1+window_size-step_size), start_2:end_2, ...] /= 2
                    if ii==(math.ceil((dimension[1]-window_size)/step_size)+1-1):
                        combining[(end_1-step_size):end_1, start_2:(dimension[1]-(dimension[1]-window_size)%step_size), ...] /= 2
                    else:
                        combining[(end_1-step_size):end_1, start_2:(start_2+window_size-step_size), ...] /= 2
            elif ii!=0 and jj==0:
                if ii==(math.ceil((dimension[1]-window_size)/step_size)+1-1):
                    combining[start_1:end_1, start_2:(dimension[1]-(dimension[1]-window_size)%step_size), ...] /= 2
                else:
                    combining[start_1:end_1, start_2:(start_2+window_size-step_size), ...] /= 2
            """
            combining[128:240, 0:128] = img[1, 16:128,0:128]
            combining[0:128, 128:240] = img[2, 0:128, 16:128]
            combining[128:240, 128:240] = img[3, 16:128, 16:128]
            """
    return combining

def view_generator(generator, model, multiview):
    

    if multiview == 0:
        dim = (240, 240, 2)
        final = np.zeros(shape=[240,240,155,2])
        final_img = np.zeros(shape=[240,240,155,2])
        remain = 155
    elif multiview == 1 or multiview == 2:
        dim = (240, 155, 2)
        final = np.zeros(shape=[240,155,240,2])
        final_img = np.zeros(shape=[240,155,240,2])
        remain = 240
    else:
        print("multiview is wrong")
        return
    
    for ii in range(remain):
        
        image = next(generator)
        #print(ii)
        """
        if ii==120:
            for jj in range(6):
                plt.imshow(image[jj,:,:,0])
                plt.show()
        """
        result = model.predict(image)[0]
        #print(ii)
        result = tf.nn.softmax(result, axis=-1)
        #print(ii)
        """
        combined_img = np.zeros(shape=[240,240,4])
        for jj in range(4):
            combined_img[:,:,jj] = combine_crop(image[:,:,:,jj], dim=dim[0:2])

        combined_seg = combine_crop(mask)
        """
        #print(ii)
        #print(result.shape)
        if ii==130:
            combined_result = combine_crop(result, dimension=dim)#, temp=True)
        else:
            combined_result = combine_crop(result, dimension=dim)
        #combined_result = create_mask(combined_result)
        #combined_result = np.squeeze(combined_result, axis = -1)
        """
        if ii==89:
            #print([mask[0,...].shape, create_mask(combined_result).shape])
            
            for jj in range(6):
                
                np.set_printoptions(threshold=np.inf)
                print(result[jj, ..., 1])
                print("hihi", result[jj, :,:, 1].max())
                display([image[jj,...,0], mask[jj,...], result[jj, :,:, 1], create_mask(combined_result)])
        """
        """
        for jj in range(result.shape[0]):
            a = image[jj, :,:,:]
            b = mask[jj, :,:]
            c = result[jj, :,:,:]
            if jj==2:
                display([a[:,:,0], a[:,:,1], a[:,:,2], a[:,:,3], b, create_mask(c)])
        """
        #if multiview==1:
            #combined_result = cv.flip(combined_result, 1)
        #print(ii)
        final[:,:,ii,:] = combined_result
        #display([combined_img[:,:,0], combined_img[:,:,1], combined_img[:,:,2], combined_img[:,:,3], combined_seg, combined_result])
        
    
    if multiview == 1:
        final = np.swapaxes(final,0,1)
        final = np.swapaxes(final,0,2)

    elif multiview == 2:
        final = np.swapaxes(final,1,2)

    return final

def moving_window_crop(img):
    window_size = 128
    step_size = 32
    a = 0
    b = 0
    while True:
        c = a + window_size
        d = b + window_size
        
        if c >= img[..., 0].shape[0]:
            c = img[..., 0].shape[0]
            a = c - window_size

        if d >= img[..., 0].shape[1]:
            d = img[..., 0].shape[1]
            b = d - window_size

        temp1 = img[a:c, b:d, :]
        yield temp1

        if c==img[..., 0].shape[0]:
            a = 0
            b = b + step_size
        else:
            a = a + step_size 
        
        if c==img[..., 0].shape[0] and d==img[..., 0].shape[1]:
            break

def data_generator(index, skip_all_zero_mask = True, skip_all_zero_image = False, skip_rate = 100, augmentation = False, cropping = False, multiview = 0):

    file_path = 'D:\BMED4010\dataset\BRATS2020\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'

    flair_paths = [
        os.path.join(file_path, x, x+"_flair.nii.gz")
        for x in os.listdir(file_path)
    ]

    t1_paths = [
        os.path.join(file_path, x, x+"_t1.nii.gz")
        for x in os.listdir(file_path)
    ]

    t2_paths = [
        os.path.join(file_path, x, x+"_t2.nii.gz")
        for x in os.listdir(file_path)
    ]

    t1ce_paths = [
        os.path.join(file_path, x, x+"_t1ce.nii.gz")
        for x in os.listdir(file_path)
    ]
    

    flair_xnames = np.take(flair_paths, index)
    t1_xnames = np.take(t1_paths, index)
    t2_xnames = np.take(t2_paths, index)
    t1ce_xnames = np.take(t1ce_paths, index)
    
    x_names = np.stack((flair_xnames, t1_xnames, t2_xnames, t1ce_xnames))

    a=0
    while a<len(flair_xnames):
        print(index[a])
        process_gen = process_both(x_names[:, a], skip_all_zero_image, skip_all_zero_mask, skip_rate, cropping, multiview=multiview)               
        for jj in range(1000):
            
            image = next(process_gen, (None, None))
            
            if np.all(image[0] == None):
                break

            image = np.swapaxes(image,0,-1)
            image = np.swapaxes(image,0,1)

            crop_gen = moving_window_crop(image)
            cropped_image = []
            
            while True:
                x_crop = next(crop_gen, (None, None))
                #print(type(x_crop), np.all(x_crop[0] == None))
                if np.all(x_crop[0] == None):
                    break
                cropped_image.append(x_crop)

            cropped_image = np.array(cropped_image)
            #print(cropped_image.shape, cropped_seg.shape)
            yield cropped_image
        
        a = (a+1)#%len(y_names)

high_val_index = [84, 336, 45, 176, 143, 125, 227, 118, 9, 90, 147,
 351, 33, 357, 221, 77, 46, 5, 231, 244, 124, 238, 355, 101, 75, 347,
  350, 109, 60, 202, 42, 56, 361, 117, 59, 165, 358, 113, 159, 63, 146,
   66, 345, 158, 181, 30, 22, 24, 353, 119, 246, 79, 17, 172, 213, 177, 183, 168, 6]
   
low_val_index = [263, 294, 269, 259, 304, 306, 324, 312, 309, 287, 326, 329, 277, 271, 317, 292]


def BraTS_val_gen(multiview):
    
    index = high_val_index+low_val_index
    
    return data_generator(index, skip_all_zero_mask = True, skip_rate = 93, augmentation = True, multiview=multiview)   

def get_distance_3D(largest_component, target_component):
    # Find the points corresponding to zeros and ones
    #zero_indices = largest_component
    #one_indices = target_component
    # Compute all pairwise distances between zero-points and one-points
    pairwise_distances = cdist(largest_component, target_component, 'euclidean')
    # Choose the minimum distance
    min_dist = np.min(pairwise_distances[pairwise_distances!=0])   
    return min_dist 


def connected_components_3D(image, threshold_min = 0, threshold_max = 256*256, threshold_dist = 240):
    labels_out, N = cc3d.connected_components(image, return_N = True)
    counter = np.bincount(labels_out.flatten())
    #print(np.bincount(labels_out.flatten()))

    distance = np.zeros(counter.shape)
    max_component_index = np.argmax(counter[1:]) + 1
    largest_component = np.zeros(image.shape)
    largest_component[labels_out==max_component_index] = 1
    largest_coord = np.transpose(np.nonzero(largest_component==1))
    for ii in range(1, N+1):
        if ii!=max_component_index:
            target_component = np.zeros(image.shape)
            target_component[labels_out==ii] = 1
            target_coord = np.transpose(np.nonzero(target_component==1))
            distance[ii] = get_distance_3D(largest_coord, target_coord)


    result = np.zeros(image.shape, dtype = np.int16)

    for jj in range(1, N+1):
        if (counter[jj]>threshold_max or counter[jj]<threshold_min) and distance[jj]<threshold_dist:
            result[labels_out==jj] = 1
    return result




model_241_axial = tf.keras.models.load_model("label241_axial_final1")
model_241_coronal = tf.keras.models.load_model("label241_coronal_final3")
model_241_sagittal = tf.keras.models.load_model("label241_sagittal_final3")

model_41_axial = tf.keras.models.load_model("label41_axial_final1")
model_41_coronal = tf.keras.models.load_model("label41_coronal_final3")
model_41_sagittal = tf.keras.models.load_model("label41_sagittal_final3")

model_4_axial = tf.keras.models.load_model("label4_axial_final1")
model_4_coronal = tf.keras.models.load_model("label4_coronal_final3")
model_4_sagittal = tf.keras.models.load_model("label4_sagittal_final3")

gen_241_axial = BraTS_val_gen(multiview=0)
gen_241_coronal = BraTS_val_gen(multiview=2)
gen_241_sagittal = BraTS_val_gen(multiview=1)

gen_41_axial = BraTS_val_gen(multiview=0)
gen_41_coronal = BraTS_val_gen(multiview=2)
gen_41_sagittal = BraTS_val_gen(multiview=1)

gen_4_axial = BraTS_val_gen(multiview=0)
gen_4_coronal = BraTS_val_gen(multiview=2)
gen_4_sagittal = BraTS_val_gen(multiview=1)

index = high_val_index+low_val_index

for ii in range(len(index)):

    file_path = 'D:\BMED4010\dataset\BRATS2020\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
    flair_paths = [
        os.path.join(file_path, x, x+"_flair.nii.gz")
        for x in os.listdir(file_path)
    ]

    scan = nib.load(flair_paths[index[ii]])
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------

    # multiview == 1 --> sagittal, 2 --> coronal
    result_241_axial = view_generator(gen_241_axial, model_241_axial, multiview=0)
    result_241_coronal = view_generator(gen_241_coronal, model_241_coronal, multiview=2)
    result_241_sagittal = view_generator(gen_241_sagittal, model_241_sagittal, multiview=1)    

    result_241 = (result_241_axial + result_241_coronal + result_241_sagittal)/3
    result_241 = create_mask(result_241)
    result_241 = connected_components_3D(result_241, threshold_min=0, threshold_max=200, threshold_dist=25)

    result_241_axial = None
    result_241_coronal = None
    result_241_sagittal = None

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    result_41_axial = view_generator(gen_41_axial, model_41_axial, multiview=0)
    result_41_coronal = view_generator(gen_41_coronal, model_41_coronal, multiview=2)
    result_41_sagittal = view_generator(gen_41_sagittal, model_41_sagittal, multiview=1)

    result_41 = (result_41_axial + result_41_coronal + result_41_coronal)/3
    result_41 = create_mask(result_41)

    result_41_axial = None
    result_41_coronal = None
    result_41_sagittal = None

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------

    result_4_axial  = view_generator(gen_4_axial, model_4_axial, multiview=0)
    result_4_coronal = view_generator(gen_4_coronal, model_4_coronal, multiview=2)
    result_4_sagittal  = view_generator(gen_4_sagittal, model_4_sagittal, multiview=1)

    result_4 = (result_4_axial + result_4_coronal + result_4_sagittal)/3
    result_4 = create_mask(result_4)

    result_4_axial = None
    result_4_coronal = None
    result_4_coronal = None

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------

    final = np.zeros((240,240,155), dtype = np.int16)
    
    final[result_241==1] = 2
    final[np.logical_and(final, result_41)==True] = 1
    final[np.logical_and(final, result_4)==True] = 4

    if (final==4).size<=200:
        final[final==4] = 1
    
    result_241 = None
    result_41 = None
    result_4 = None

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    volume = nib.Nifti1Image(final, [[-1,0,0,0],[0,-1,0,239],[0,0,1,0],[0,0,0,1]])
    volume.header['regular'] = scan.header['regular']
    volume.header['pixdim'] = scan.header['pixdim']
    volume.header['qform_code'] = scan.header['qform_code']
    volume.header['sform_code'] = scan.header['sform_code']

    
    file_path = "D:\\BMED4010\\upload validation\\05 final 3 validation set version\\"
    ID = str(index[ii]+1).zfill(3)
    nib.save(volume, file_path+"BraTS20_Training_"+ID+".nii.gz")

    final = None

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

a = BraTS_val_gen(0)

b = next(a)
print('hi')
print(type(b))
print(b.shape)

"""