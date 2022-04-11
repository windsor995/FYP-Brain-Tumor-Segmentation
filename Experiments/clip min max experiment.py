# multimodality reason example: image 10 (BraTS20_Training_010), slice 91.
# "D:\BMED4010\model\\09 U-net test 7\\training_1\cp-0009.ckpt" acceptable performance (let Vick see see)
# "D:\BMED4010\model\\10 U-net test 8\\training_3\cp-0008.ckpt" or any other things in training 2 are ok. Pick some see see later
#229/229 - 159s - loss: 0.0854 - accuracy: 0.9876 - 159s/epoch - 693ms/step [0.08543605357408524, 0.9875937700271606]
#mask --> 0, 2, 4, 1 (0: background, 1: tumour core)


import os
from re import S, X
import zipfile
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import nibabel as nib
import SimpleITK as sitk
import math

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

import cv2 as cv
import random
import datetime
"""
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
"""
def read_nifti_file(filepath):
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def min_max_rescale(volume):

    if np.all(volume==0):
        return volume
    
    volume = volume.astype("float64")
    min = volume.min()
    max = volume.max()
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    return volume

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
        temp = img[ii,...]
        if np.all(temp==0):
            continue
        up, low = np.percentile(temp[temp>0], 99), np.percentile(temp[temp>0], 1)
        temp[np.logical_and(temp>up, temp>0)] = up
        temp[np.logical_and(temp<low,temp>0)] = low
        img[ii,...] = temp
    return img
        
def rand_crop(img, seg):
    crop_row, crop_column = random.randint(20,210-128), random.randint(0,150-128)    
    return img[crop_row:crop_row+128, crop_column:crop_column+128, :], seg[crop_row:crop_row+128, crop_column:crop_column+128] 

def moving_window_crop(img, seg):

    window_size = 128
    step_size = 64

    a = 0
    b = 0

    while True:
        c = a + window_size
        d = b + window_size
        
        if c >= seg.shape[0]:
            c = seg.shape[0]
            a = c - window_size

        if d >= seg.shape[1]:
            d = seg.shape[1]
            b = d - window_size

        #print(a,b,c,d)

        temp1 = img[a:c, b:d, :]
        temp2 = seg[a:c, b:d]
        yield temp1, temp2

        if c==seg.shape[0]:
            a = 0
            b = b + step_size
        else:
            a = a + step_size
            
        if c==seg.shape[0] and d==seg.shape[1]:
            break
        
def classiffication(volume):
    volume[volume==1]=0
    volume[volume==2]=0
    volume[volume==4]=1

    return volume

def t1_estimation_on_t2(img):
    temp = img[1, ...].copy()
    if np.all(temp==0):
        return img
    
    maximum = temp.max()
    minimum = 0
    middle = (maximum-minimum)/2

    temp[temp!=0] = (temp[temp!=0]-middle)*-1+middle

    img[1, ...] = np.clip(temp, a_min = 0., a_max = temp.max())
    return img
    

def process_both(xpath, ypath, xskip, yskip, skip_rate, cropping, skip_size):
    xvolume = []
    for ii in xpath:
        xvolume.append(read_nifti_file(ii))
    xvolume = np.array(xvolume)
    xvolume = np.swapaxes(xvolume,1,-1)
    xvolume = np.swapaxes(xvolume,1,2) 
    
    yvolume = read_nifti_file(ypath)
    yvolume = np.swapaxes(yvolume,0,-1)
    yvolume = np.swapaxes(yvolume,0,1)

    for ii in range(xvolume.shape[-1]-1, -1, -1):      
                
        yone_slice = yvolume[:,:,ii]    
        yone_slice = classiffication(yone_slice)
        yone_slice = yone_slice.astype(np.int32)

        if yskip and ((yone_slice[yone_slice>0].size<=skip_size and not np.all(yone_slice==0)) or (np.all(yone_slice==0) and random.randint(1,100)<=skip_rate)): #
            continue

        xone_slice = xvolume[:,:,:,ii]
        if xskip and np.all(xone_slice==0) and random.randint(1,100)<=skip_rate:
            continue
       
        xone_slice = percentile_clipping(xone_slice)
        xone_slice = min_max_rescale(xone_slice)
        
        #xone_slice = t1_estimation_on_t2(xone_slice)
        #print(xone_slice[1,...].min())

        #xone_slice = normalize(xone_slice)

        yield xone_slice, yone_slice
    

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
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def tf_get_sum(thing):
    temp = tf.cast(thing, tf.float64)
    temp = tf.reduce_sum(temp)
    return temp


def dice_coef_2cat(y_true, y_pred, smooth=1e-7):
    
    
    total_class = 2
    except_last = [1,2,3]
    
    y_true_f = tf.cast(y_true, tf.float64)
    y_pred_f = tf.cast(tf.nn.softmax(y_pred, axis=-1), tf.float64)
    
    #y_true_f = tf.cast(K.one_hot(K.cast(tf.squeeze(y_true, axis = [-1]), 'int32'), num_classes=total_class), dtype = tf.float64)
    #y_pred_f = y_pred_f
    
    intersect = 2.*K.sum(y_true_f * y_pred_f, axis = [1,2])
    denom = K.sum(y_true_f + y_pred_f, axis = [1,2])

    dice = 0.5*K.sum(((intersect+smooth)/(denom+smooth)), axis = -1)
    
    return K.mean(dice)

    """
    #print((y_true), (y_pred))
    total_class = 2
        
    y_true_f = tf.cast(y_true, dtype = tf.float64)
    y_pred_f = tf.cast(y_pred, dtype = tf.float64)

    class_count = tf.cast(tf.reduce_sum(y_true_f, axis = [1,2]), dtype=tf.float64)
    class_weights = tf.cast(tf.constant([0.2, 0.8]), dtype=tf.float64)/(class_count)#1./(class_count)
    #print(class_count.shape, class_weights.shape)
    class_weights = tf.where(tf.math.is_finite(class_weights), class_weights, 0.2)#4e-4

    
    intersect = y_true_f * y_pred_f
    intersect = class_weights*K.sum(intersect, axis=[1,2])
    intersect = K.sum(intersect, axis = -1)

    denom = y_true_f + y_pred_f
    denom = class_weights*K.sum(denom, axis=[1,2])
    denom = K.sum(denom, axis = -1)
    
    #print(class_count, class_weights, intersect, denom)

    dice = 2. * intersect / (denom+smooth)
    dice = tf.where(tf.math.is_finite(dice), dice, 0.)
    #print(K.mean(dice).shape)
    return K.mean(dice)
    """

def cross_entropy(y_true, y_pred, pos_weight, smooth=1e-7):

    

    #thing = keras.losses.BinaryCrossentropy()#reduction = tf.keras.losses.Reduction.NONE)
    #entropy = thing(y_true, y_pred)
    
    #weight_vector = y_true*weights[1] + (1-y_true)*weights[0]
    #final = weight_vector*entropy
    
    y_true_f = tf.cast(tf.math.argmax(y_true, axis=-1), dtype = tf.float64)
    y_pred_f = tf.cast(tf.gather(tf.nn.softmax(y_pred, axis=-1), tf.cast(y_true_f, dtype = tf.int64), axis=-1, batch_dims=3), dtype=tf.float64)

    
    y_pred_f = tf.cast(tf.clip_by_value(y_pred_f, clip_value_min=np.nextafter(0., 1.), clip_value_max=np.nextafter(1., 0.)), dtype=tf.float64)

    entropy = y_true_f*-1*tf.math.log(y_pred_f)
    entropy += (1.-y_true_f)*-1*tf.math.log(y_pred_f)*pos_weight
    

    #y_true_f = tf.cast(y_true, dtype = tf.float64)
    #y_pred_f = tf.cast(y_pred, dtype = tf.float64)
    
    #entropy = tf.nn.weighted_cross_entropy_with_logits(y_true_f, y_pred_f, pos_weight)
    
    entropy = tf.math.reduce_mean(entropy, axis = [1,2])
    entropy = tf.math.reduce_mean(entropy, axis = [0])
    
    return entropy

def dice_coef_2cat_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_2cat(y_true, y_pred)

def dice_entropy_loss(y_true, y_pred):
    
    total_class = 2
        
    #y_true_f = tf.cast(y_true, dtype = tf.float64)
    #y_pred_f = tf.cast(y_pred, dtype = tf.float64)
    """
    class_count = tf.cast(tf.reduce_sum(y_true_f, axis = [1,2], keepdims = True), dtype=tf.float64)
    class_weights = tf.cast(tf.constant([0.5, 0.5]), dtype=tf.float64)/(class_count)#1./(class_count)
    class_weights = tf.where(tf.math.is_finite(class_weights), class_weights, 0.2)#4e-4
    """
    
    #entropy = cross_entropy(y_true, y_pred, tf.constant([1.7], dtype=tf.float64))

    dice = dice_coef_2cat_loss(y_true, y_pred)

    #print(entropy.shape, dice.shape, (entropy+dice).shape)
    return dice#0.9*entropy+0.1*dice


def BRATS_dice_coef(y_true, y_pred, smooth = 1e-7):
    
    total_class = 2
    except_last = [1,2]
    
    y_true_f = tf.cast(y_true, tf.float64)
    y_pred_f = tf.cast(tf.nn.softmax(y_pred, axis=-1), tf.float64)
    
    #y_true_f = tf.cast(K.batch_flatten(K.one_hot(K.cast(tf.squeeze(y_true_f, axis = [-1]), 'int64'), num_classes=total_class)[..., 1:]), dtype = tf.float64)
    y_true_f = tf.cast(K.batch_flatten(y_true_f[..., 1:]), dtype = tf.float64)
    y_pred_f = K.batch_flatten(y_pred_f[..., 1:])
    
    intersect = K.sum(y_true_f * y_pred_f, axis = -1)
    denom = K.sum(y_true_f + y_pred_f, axis = -1)
    return K.mean((2.*intersect/(denom+smooth)))

def BRATS_dice_loss(y_true, y_pred, smooth = 1e-7):
     return 1 - BRATS_dice_coef(y_true, y_pred)
    
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    flair_val.show_predictions(model)
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def history_of_model(model_history, lim = False):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure()
    for ii in list(model_history.history):
        plt.plot(model_history.epoch, model_history.history[ii], label=ii)
        #plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training Loss, Validation Loss, and Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    if lim != False:
        plt.ylim([0, 1])
    plt.legend()
    plt.show()

def show_predictions(data, slice_num = [0], batch_num = 1):
    
    for a, b in data.take(batch_num):
        a = np.array(a)
        #b = np.array(b[0])
        #print(b[0].shape, b[1].shape, b[2].shape, b[3].shape)
        for ii in slice_num:

            image = a[ii, :,:,:]
            
            true_mask = [temp[ii,:,:,:] for temp in b]
            true_mask_0 = true_mask[0]
            
            
            result = model.predict(image[np.newaxis, ...])
            result = result[0]
            #print(weighted_categorical_crossentropy(true_mask[np.newaxis, ...], result).numpy())#dice_coef_2cat(
            model.evaluate(x = image[np.newaxis,...], y = [temp[np.newaxis,...] for temp in true_mask], verbose = 2)
            #print(BRATS_dice_coef(true_mask_0[np.newaxis,...], result))
            
            result = create_mask(result)
            result = np.array(result)

            """
            result = union(image)
            result = result[:,:,0]
            """
            #print('hi',true_mask[0].shape)
            display([image[:,:,0], image[:,:,1], image[:,:,2], image[:,:,3], create_mask(true_mask[0]), result])
    

def view_generator(generator):
    for ii in range(60):
        image, mask = next(generator)
        image = np.array(image)
        mask = np.array(mask[0])

        print(ii)
        
        display([image[:,:,0], image[:,:,1], image[:,:,2], image[:,:,3], mask, create_mask(model.predict(image[np.newaxis,...])[0])])

def flipping(img, seg, prob):
    if random.randint(1,100)<=prob:
        temp = random.randint(-1,1)
        img = cv.flip(img, temp)
        seg = cv.flip(seg, temp)
    return img, seg

def rotating(img, seg, prob):
    if random.randint(1,100)<=prob:
        """
        temp = random.randint(1,359)
        img = ndimage.rotate(img, temp, reshape = False, prefilter = False, order = 0)
        seg = ndimage.rotate(seg, temp, reshape = False, prefilter = False, order = 0)
        """
        temp = random.randint(1,3)
        img = ndimage.rotate(img, temp*90, reshape = False)
        seg = ndimage.rotate(seg, temp*90, reshape = False)
        
    return img, seg

def gaussian_blur(img):
    sigma = np.random.uniform(0.1, 0.5)
    for ii in range(4):
        img[...,ii] = ndimage.filters.gaussian_filter(img[...,ii], sigma)
    return img

def contrast_aug(img, prob):
    if random.randint(1,100)<=prob:
        value = np.random.uniform(0.8, 1.2)
        for ii in range(4):
            temp = img[:,:,ii]
            if np.all(temp==0):
                continue
            temp[temp!=0] = (temp[temp!=0]-np.mean(temp[temp!=0], dtype = np.float64))*value+np.mean(temp[temp!=0], dtype = np.float64)
            img[:,:,ii] = temp
    return img

def scaling(img, seg, prob): #resize and then do zero padding or cropping
    if np.all(img==0):
        return img, seg
    if random.randint(1,100)<=prob:
        width = img.shape[0]
        height = img.shape[1]

        new_width = int(round(width*np.random.uniform(0.9, 1.2), 0))
        new_height = int(round(height*np.random.uniform(0.9, 1.2), 0))

        new_img = cv.resize(img, (new_height, new_width), interpolation = cv.INTER_NEAREST)
        new_seg = cv.resize(seg, (new_height, new_width), interpolation = cv.INTER_NEAREST)

        #print(new_img.shape, new_seg.shape)
        #print(width, height, new_width, new_height)
        if width>new_width:
            diff = width-new_width
            new_img = np.pad(new_img, [(diff//2, diff-diff//2), (0,0), (0,0)])
            new_seg = np.pad(new_seg, [(diff//2, diff-diff//2), (0,0)])
            
        elif width<new_width:
            new_img = new_img[new_width//2-(width//2):new_width//2+width-width//2,:,:]
            new_seg = new_seg[new_width//2-(width//2):new_width//2+width-width//2,:]

        if height>new_height:
            diff = height-new_height
            new_img = np.pad(new_img, [(0,0), (diff//2, diff-diff//2), (0,0)])
            new_seg = np.pad(new_seg, [(0,0), (diff//2, diff-diff//2)])
            
        elif height<new_height:
            new_img = new_img[:,new_height//2-(height//2):new_height//2+height-height//2,:]
            new_seg = new_seg[:,new_height//2-(height//2):new_height//2+height-height//2]
        #print(new_img.shape, new_seg.shape)

        img, seg = new_img, new_seg
    return img, seg

def elastic_transform(img, seg, alpha, sigma):
    a = np.random.uniform(alpha[0], alpha[1])
    s = np.random.uniform(sigma[0], sigma[1])

    shape = (img.shape[0], img.shape[1])
    temp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*temp, indexing='ij')).astype(np.float64)
    #print(temp, *temp,coords.shape)
    #for d in range(len(shape)):
        #coords[d] -= ((np.array(shape).astype(np.float64)-1)/2.)[d]
    #print(coords)
    n_dim = len(coords)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            ndimage.filters.gaussian_filter((np.random.random(coords.shape[1:]) * 2 - 1), s, mode="constant", cval=0) * a)
    offsets = np.array(offsets)
    coords = offsets + coords

    for ii in range(img.shape[-1]):
        img[:,:,ii] = ndimage.map_coordinates(img[:,:,ii], coords, order=3, mode='nearest', cval=0)
    seg = ndimage.map_coordinates(seg, coords, order=3, mode='nearest', cval=0)
    #np.set_printoptions(threshold=np.inf)
    #print(seg)
    #os.system("pause")
    return img, seg


def data_augmentation1(img, seg):

    prob1 = 40
    prob2 = 16
    prob3 = 30
    prob4 = 20

    have_seg = not np.all(seg==0)
    
    if np.all(img==0):
        if random.randint(1,100)<=70:
            img, seg = rand_crop(img, seg)
        else: #simulation of low resolution
            img = cv.resize(img, (128,128), interpolation = cv.INTER_NEAREST)
            seg = cv.resize(seg, (128,128), interpolation = cv.INTER_NEAREST)
        return img, seg    
    
    if random.randint(1,100)<=25 and have_seg:#25
        #print("before", seg.max())
        img, seg = elastic_transform(img, seg, (0.,230.), (12.,16.))
        #print("after", seg.max())
    
    img, seg = rotating(img, seg, prob1)

    img, seg = scaling(img, seg, prob1)
    
    if random.randint(1,100)<=70:
        img, seg = rand_crop(img, seg)
    else: #simulation of low resolution
        img = cv.resize(img, (128,128), interpolation = cv.INTER_NEAREST)
        seg = cv.resize(seg, (128,128), interpolation = cv.INTER_NEAREST)
    
    if random.randint(1,100)<=prob4:
        img = gaussian_blur(img)
    
    img, seg = flipping(img, seg, prob1)

    #img = contrast_aug(img, prob4)

    
    if random.randint(1,100)<=prob4:
        for ii in range(4):
            img[:,:,ii] = img[:,:,ii] + np.random.normal(0, 0.1)
    
    """   
    if random.randint(1,100)<=prob3:
        #gamma correction
        img = np.sign(img) * (np.abs(img)**random.uniform(0.8, 1.2)) #, size = image.shape
    """
    if random.randint(1,100)<=prob3:
        img = img + np.random.normal(0, random.uniform(0,0.1), img.shape)
     
    """
    if random.randint(1,100)<=prob2:
        img[:,:,random.randint(0,3)] = 0
    """
    return img, seg

def data_generator(index, skip_all_zero_mask = True, skip_all_zero_image = False, skip_rate = 100, augmentation = False, cropping = False, skip_size=5):

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
    
    seg_paths = [
        os.path.join(file_path, x, x+"_seg.nii.gz")
        for x in os.listdir(file_path)
    ]

    flair_xnames = np.take(flair_paths, index)
    t1_xnames = np.take(t1_paths, index)
    t2_xnames = np.take(t2_paths, index)
    t1ce_xnames = np.take(t1ce_paths, index)
    y_names = np.take(seg_paths, index)
    
    flair_xnames, t1_xnames, t2_xnames, t1ce_xnames, y_names = shuffle(flair_xnames, t1_xnames, t2_xnames, t1ce_xnames, y_names)
    x_names = np.stack((flair_xnames, t1_xnames, t2_xnames, t1ce_xnames))

    a=0
    while a<len(y_names):
        #print(flair_xnames[a])
        #display([c[:,:,0], c[:,:,1], c[:,:,2], c[:,:,3],b[1]])

        process_gen = process_both(x_names[:, a], y_names[a], skip_all_zero_image, skip_all_zero_mask, skip_rate, cropping, skip_size)
        #scan_gen = process_scan_multi(x_names[:, a], skip_all_zero_image, skip_rate)
        #seg_gen = process_seg(y_names[a], skip_all_zero_mask, skip_rate)
        
        
        for jj in range(1000):
            
            image, seg = next(process_gen, (None, None))

            if np.all(image == None):
                break
                     
            
            """          
                if (np.all(seg==0) or seg[seg>0].size<50) and skip_all_zero_mask and random.randint(1,100)<=skip_rate:
                    continue
                          
                if np.all(image==0) and skip_all_zero_image and random.randint(1,100)<=skip_rate:
                    continue
            """
            image = np.swapaxes(image,0,-1)
            image = np.swapaxes(image,0,1)

            if augmentation:
                image, seg = data_augmentation1(image, seg)
                #print(type(image), type(seg))
                """
                crop_gen = moving_window_crop(image, seg)
                while True:
                    image, seg = next(crop_gen, (None, None))
                    if np.all(image==None):
                        break
                    yield image, seg[..., np.newaxis]
                """
            else:
                #image, seg = rand_crop(image, seg)
                image = cv.resize(image, (128,128), interpolation = cv.INTER_NEAREST)
                seg = cv.resize(seg, (128,128), interpolation = cv.INTER_NEAREST)
                #print(type(image), type(seg))
                #print("hi", seg, seg.shape)
                #print(image.shape, seg.shape)

            #print(a, jj)
            
            if (not np.all(seg==0)) and seg[seg>0].size<=skip_size and skip_all_zero_mask:
                continue
            
        
            seg8 = cv.resize(seg, (64,64), interpolation = cv.INTER_NEAREST)
            seg7 = cv.resize(seg, (32,32), interpolation = cv.INTER_NEAREST)
            #seg6 = cv.resize(seg, (16,16), interpolation = cv.INTER_NEAREST)
            #print("hihihihihih", type(seg6[..., np.newaxis]))
            
            
            seg = tf.cast(K.one_hot(K.cast(seg, 'int32'), num_classes=2), dtype = tf.int32)

            
            seg8 = tf.cast(K.one_hot(K.cast(seg8, 'int32'), num_classes=2), dtype = tf.int32)
            seg7 = tf.cast(K.one_hot(K.cast(seg7, 'int32'), num_classes=2), dtype = tf.int32)
            #seg6 = tf.cast(K.one_hot(K.cast(seg6, 'int32'), num_classes=2), dtype = tf.int32)

            
            yield (image, (seg, seg8, seg7))
        
        a = (a+1)#%len(y_names)

all_index = list(range(369))
high_index = all_index[0:259] + all_index[335:369]
low_index = all_index[259:335]

high_train_index, high_val_index = train_test_split(high_index, test_size = 0.2, random_state=42)
low_train_index, low_val_index = train_test_split(low_index, test_size = 0.2, random_state=42)

#print(len(high_train_index), len(low_train_index))
#print(len(high_val_index), len(low_val_index))

def flair_train_gen():

    #label 241: (high: 15091, Low: 3789) lable 41: 70: (high: 21689 Low:5801) 80: (high: 15197 Low: 4005) 90: (high: 14775 Low:4128) label 4: 80: (high: 17868 Low: 3190)
    index = high_train_index + low_train_index + low_train_index + low_train_index + low_train_index 
    
    return data_generator(index, skip_all_zero_mask = True, skip_all_zero_image = True, skip_rate = 80, augmentation = True, skip_size=3)   #tumor4: 84


def flair_val_gen():

    #label 241: (both: 4674, high: 3831, low: 1012) lable 41: 70: (high: 5344 Low:1560) 80: (high: 3697 Low: 1172) 90:(high: 3570 Low:1186) label 4: 80: (high: 4362 Low: 902)
    index = high_val_index + low_val_index
    
    return data_generator(index, skip_size=10)

def _fixup_shape(images, labels):
    total_class = 2
    images.set_shape([128, 128, 4])
    labels[0].set_shape([128, 128, total_class])
    labels[1].set_shape([64, 64, total_class])
    labels[2].set_shape([32, 32, total_class])
    #labels[3].set_shape([16, 16, total_class])
    return images, labels

BATCH_SIZE = 50
BUFFER_SIZE = 1200
EPOCHS = 200
STEPS_PER_EPOCH = 32000//BATCH_SIZE#250 #1000#1280#250 #1000 #335 #tumor 241: 494   # high low grade same 251   #533  # low grade tumour: 109 #17072



file_path = 'D:\BMED4010\dataset\BRATS2020\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'

flair_paths = [
    os.path.join(file_path, x, x+"_flair.nii.gz")
    for x in os.listdir(file_path)
]

seg_paths = [
    os.path.join(file_path, x, x+"_seg.nii.gz")
    for x in os.listdir(file_path)
]

flair_train = tf.data.Dataset.from_generator(flair_train_gen, output_types = (tf.float64, (tf.int32, tf.int32, tf.int32)))
flair_train = flair_train.map(_fixup_shape)
flair_train = flair_train.repeat()
flair_train = flair_train.shuffle(buffer_size = BUFFER_SIZE)
flair_train = flair_train.batch(batch_size = BATCH_SIZE)
flair_train = flair_train.prefetch(buffer_size=tf.data.AUTOTUNE)


flair_val = tf.data.Dataset.from_generator(flair_val_gen, output_types = (tf.float64, (tf.int32, tf.int32, tf.int32)))
flair_val = flair_val.map(_fixup_shape)
flair_val = flair_val.repeat()
flair_val = flair_val.shuffle(buffer_size = BUFFER_SIZE)
flair_val = flair_val.batch(batch_size = BATCH_SIZE)
flair_val = flair_val.prefetch(buffer_size=tf.data.AUTOTUNE)




#---------------------------------------------------------------------------------------------------------------------

def high_val_gen():
    
    index = high_val_index

    return data_generator(index, skip_size=3)
    
high_val = tf.data.Dataset.from_generator(high_val_gen, output_types = (tf.float64, (tf.int32, tf.int32, tf.int32)))
high_val = high_val.map(_fixup_shape)
high_val = high_val.repeat()
high_val = high_val.shuffle(buffer_size = BUFFER_SIZE)
high_val = high_val.batch(batch_size = BATCH_SIZE)
high_val = high_val.prefetch(buffer_size=tf.data.AUTOTUNE)

def low_val_gen():
    
    index = low_val_index

    return data_generator(index, skip_size=3)

low_val = tf.data.Dataset.from_generator(low_val_gen, output_types = (tf.float64, (tf.int32, tf.int32, tf.int32)))
low_val = low_val.map(_fixup_shape)
low_val = low_val.repeat()
low_val = low_val.shuffle(buffer_size = BUFFER_SIZE)
low_val = low_val.batch(batch_size = BATCH_SIZE)
low_val = low_val.prefetch(buffer_size=tf.data.AUTOTUNE)

def conv_block(block_name, inputs, channel, kernel, stride=1, activation=True):
    with K.name_scope(block_name):     
        conv = layers.Conv2D(channel, kernel, padding = 'same', strides = stride, use_bias=False)(inputs)#, kernel_regularizer=regularizers.l2(0.00001)
        conv = layers.BatchNormalization()(conv)
        #conv = tfa.layers.GroupNormalization(groups=16)(conv)#channel//32
        if activation==False:
            return conv
        #conv = tf.keras.layers.LeakyReLU(alpha=leak)(conv)
        conv = tf.keras.layers.ReLU()(conv)
    return conv

def deconvolution_block(block_name, inputs, channel, kernel, stride=2):
    with K.name_scope(block_name):
        conv = layers.Conv2DTranspose(channel, kernel, strides = stride, use_bias=False)(inputs)
        conv = layers.BatchNormalization()(conv)
        #conv = tfa.layers.GroupNormalization(groups=16)(conv)#channel//1
        #conv = tf.keras.layers.LeakyReLU(alpha=leak)(conv)
        conv = tf.keras.layers.ReLU()(conv)
    return conv

def channel_attention(block_name, inputs, ratio=16):
    with K.name_scope(block_name):
        shape = inputs.shape[1:3]
        channel = inputs.shape[-1]
        max_pool = layers.MaxPooling2D(pool_size=shape)(inputs)
        avg_pool = layers.AveragePooling2D(pool_size=shape)(inputs)

        FC = tf.keras.Sequential()
        FC.add(layers.Conv2D(channel//ratio, 1))#, kernel_regularizer=regularizers.l2(0.001), use_bias=False
        FC.add(layers.Activation("relu"))
        FC.add(layers.Conv2D(channel, 1))#, kernel_regularizer=regularizers.l2(0.001), use_bias=False

        max_out = FC(max_pool)
        avg_out = FC(avg_pool)
        out = layers.add([max_out, avg_out])
        out = layers.Activation("sigmoid")(out)
        
        return out
        
def spatial_attention(block_name, inputs, kernel=7):
    with K.name_scope(block_name):
        max_in = tf.math.reduce_max(inputs, axis=-1, keepdims=True)
        avg_in = tf.math.reduce_mean(inputs, axis=-1, keepdims=True)

        together = layers.concatenate([max_in, avg_in], axis=-1)
        out = layers.Conv2D(1, kernel, padding='same')(together)#, kernel_regularizer=regularizers.l2(0.001), use_bias=False
        out = layers.Activation("sigmoid")(out)

        return out

def CBAM(block_name, inputs):
    with K.name_scope(block_name):
        
        channel = channel_attention("channel_attention", inputs)
        channel = layers.multiply([inputs, channel])

        spatial = spatial_attention("spatial_attention", channel)
        spatial = layers.multiply([channel, spatial])
        
        return spatial

def main_block(inputs, channel, ysCBAM):
    conv = inputs
    residual = conv    

    if ysCBAM:
        conv=CBAM("attention", conv)
        conv = layers.add([conv, residual])
        
    conv = conv_block('down_1.2', conv, channel, 3)
    conv = conv_block('down_1.3', conv, channel, 3)
    conv = layers.add([conv, residual])
    #conv = tf.keras.layers.ReLU()(conv)

    return conv

leak = 0.3



def get_model(img_size, num_classes):
    
    inputs = keras.Input(shape=img_size + (4,), dtype = tf.float64)

    channel = 32
    conv1 = conv_block('down_1.1', inputs, channel, 1)
    conv1 = main_block(conv1, channel, ysCBAM=False)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    channel = 64
    conv2 = conv_block('down_2.1', pool1, channel, 1)
    conv2 = main_block(conv2, channel, ysCBAM=False)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    channel = 128    
    conv3 = conv_block('down_3.1', pool2, channel, 1)
    conv3 = main_block(conv3, channel, ysCBAM=False)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    channel = 256
    conv4 = conv_block('down_4.1', pool3, channel, 1)
    conv4 = main_block(conv4, channel, ysCBAM=False)
    #drop4 = conv4#layers.Dropout(0.5)(conv4) ################################################################################################  There is a drop out here
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)    

    channel = 320
    conv5 = conv_block('bottom1', pool4, channel, 1)
    conv5 = main_block(conv5, channel, ysCBAM=False)
    #drop5 = conv5#layers.Dropout(0.5)(conv5) ################################################################################################  There is a drop out here
    """
    pool5 = layers.MaxPooling2D(pool_size=(2, 2))(conv5)

    channel = 320
    conv6 = conv_block('bottom1', pool5, channel, 1)
    conv6 = main_block(conv6, channel, ysCBAM=False)

    channel = 256
    up5 = layers.UpSampling2D(size = (2,2))(conv6)
    up5 = conv_block('upsample6', up5, channel, 1)    
    merge6 = layers.concatenate([conv5,up5], axis = -1)
    conv6 = conv_block('up_6.1', merge6, channel, 1)
    conv6 = main_block(conv6, channel, ysCBAM=True)
    """
    channel = 256
    up5 = layers.UpSampling2D(size = (2,2))(conv5)
    up5 = conv_block('upsample5', up5, channel, 1)    
    merge6 = layers.concatenate([conv4,up5], axis = -1)
    conv6 = conv_block('up_6.1', merge6, channel, 1)
    conv6 = main_block(conv6, channel, ysCBAM=True)

    channel = 128
    up6 = layers.UpSampling2D(size = (2,2))(conv6)
    up6 = conv_block('upsample6', up6, 256, 1)    
    merge7 = layers.concatenate([conv3,up6], axis = -1)
    conv7 = conv_block('up_7.1', merge7, channel, 1)
    conv7 = main_block(conv7, channel, ysCBAM=True)

    channel = 64
    up7 = layers.UpSampling2D(size = (2,2))(conv7)
    up7 = conv_block('upsample7', up7, channel, 1)    
    merge8 = layers.concatenate([conv2,up7], axis = -1)
    conv8 = conv_block('up_8.1', merge8, channel, 1)
    conv8 = main_block(conv8, channel, ysCBAM=True)

    channel = 32
    up8 = layers.UpSampling2D(size = (2,2))(conv8)
    up8 = conv_block('upsample8', up8, channel, 1)    
    merge9 = layers.concatenate([conv1,up8], axis = -1)
    conv9 = conv_block('up_9.1', merge9, channel, 1)
    conv9 = main_block(conv9, channel, ysCBAM=True)
    
    conv10 = conv9#conv_block('end', conv9, 32, 3)
    conv10 = layers.Conv2D(num_classes, 1, name = "out128")(conv10) #, activation = 'softmax'
    
    #out6 = conv6#conv_block('end', conv6, 32, 3)
    #out6 = layers.UpSampling2D(size = (8,8))(out6)
    #out6 = layers.Conv2D(num_classes, 1, name = "out16")(out6)
    
    
    out7 = conv7#conv_block('end', conv7, 32, 3)
    #out7 = layers.UpSampling2D(size = (4,4))(out7)
    out7 = layers.Conv2D(num_classes, 1, name = "out32")(out7)
    
    out8 = conv8#conv_block('end', conv8, 32, 3)
    #out8 = layers.UpSampling2D(size = (2,2))(out8)
    out8 = layers.Conv2D(num_classes, 1, name = "out64")(out8)
    
    
    
    model = keras.Model(inputs, (conv10, out8, out7))
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model     tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.losses.Reduction.NONE)
model = get_model((128,128), 2)
"""
class scheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, end_learning_rate=0.000001, power=0.9, warm_up=False, warm_up_steps=5000):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.warm_up = warm_up
        self.warm_up_steps = warm_up_steps

    def __call__(self, step):          

        
        self.decay_steps = self.decay_steps*(step/self.decay_steps)        
        learning_rate = (self.initial_learning_rate-self.end_learning_rate) * (1 - step/self.decay_steps)**self.power + self.end_learning_rate

        if self.warm_up:
            warm_lr = (self.initial_learning_rate-self.end_learning_rate) * (1 - self.warm_up_steps/self.decay_steps)**self.power + self.end_learning_rate
            warm_lr = warm_lr*step/self.warm_up_steps
            return tf.cond(step<self.warm_up_steps, lambda:warm_lr, lambda: learning_rate)    
        return learning_rate


my_schedual = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate = 0.01, decay_steps = 100*STEPS_PER_EPOCH, end_learning_rate = 0.000001, power = 0.9)
"""
"""
test =[]

for ii in range(100000):
    test.append(my_schedual(ii))
    #print(my_schedual(ii))
plt.plot(list(range(100000)), test)
plt.show()
"""

my_schedual = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, verbose=2, min_lr = 0.50000e-05, min_delta = 0.001)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001), 
              loss=dice_entropy_loss,
              loss_weights = [4/7,2/7,1/7], 
              metrics = [BRATS_dice_coef]) #tfa.optimizers.AdamW(learning_rate = 0.0001, weight_decay=1e-4), tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.99, nesterov=True),
#tf.keras.optimizers.SGD(learning_rate=my_schedual , momentum=0.99, nesterov=True), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()
#model = tf.keras.models.load_model('D:\BMED4010\\upload validation\\01 test\\label4')

VAL_SUBSPLITS = 2
VALIDATION_STEPS = 5264//BATCH_SIZE#4000#6000#12000 #tumor241: 394 # high low grade same 513 #229 # low grade tumour: 46 #7352

root_path = "D:\\BMED4010\\model\\27 experiments\\02 clip and min max\\clip_min_max_4\\"

save_path = root_path+"cp-0018.ckpt" 
checkpoint_path = root_path+"cp-{epoch:04d}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True)

log_dir = root_path+"logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=40, write_graph=True, write_images = True, update_freq = 300, profile_batch=5)

#model.load_weights(save_path)
#model.load_weights(save_path.format(epoch=0))
#model.summary()

model_history = model.fit(flair_train, epochs=EPOCHS, 
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=flair_val,
                          callbacks = [cp_callback, my_schedual, tensorboard_callback],
                          verbose=2)

#view_generator(flair_train_gen())
#view_generator(flair_val_gen())
#model.evaluate(flair_train, steps = 500, verbose = 2)
#model.evaluate(flair_val, steps=200, verbose=2, batch_size = 16)
#model.evaluate(high_val, steps=250, verbose=2)
#model.evaluate(low_val, steps=250, verbose=2)
#show_predictions(flair_val, batch_num = 20)
#show_predictions(low_val, batch_num = 20)
#show_predictions(flair_train, range(10), batch_num=8)

#tf.keras.utils.plot_model(model, to_file = 'temp.png', show_shapes=True)

"""180289, 5116
a = flair_train_gen()
b = next(a)
c = b[0]
"""
"""
temp = 0
a = flair_train_gen()
while True:
    temp = temp+1
    print(temp)
    b = next(a)
"""


#print(BRATS_dice_coef(np.array([0,0]), np.array([[1,0], [0,1]])))
#model.save('D:\BMED4010\\upload validation\\01 test\\label4_11')

#python -m tensorboard.main --logdir=logs/fit/
