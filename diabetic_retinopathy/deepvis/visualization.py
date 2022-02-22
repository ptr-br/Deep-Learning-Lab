import logging

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import os
import gin
import pandas as pd
from matplotlib import pyplot as plt

from deepvis.deepvis import GuidedBackPropagation, GradCam, guided_grad_cam, IntegratedGradients
from input_pipeline.make_tfrecords import preprocess_image


def get_img_array(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_array = preprocess_image(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)


def plot(ax, grad_map, name):
    # grad_map = cv2.cvtColor(grad_map.astype("uint8"), cv2.COLOR_BGR2RGB)
    ax.imshow(grad_map)
    ax.set_title(name)
    ax.axis('off')


def float_2u8_norm(img):
    """ map float in any range to uint8 in range [0,255]"""
    return np.uint8(((img-img.min()) / (img.max()-img.min()+1e-7)) * 255.0)


def plot_deepvis_outputs(image,
                         guided_backprop,
                         grad_cam,
                         guided_grad_cam,
                         integrated_grad,
                         r_grade,
                         pred,
                         name,
                         cmap=None,
                         overlay_alpha=0.4):

    # make sure all inputs have correct ranges
    image = float_2u8_norm(image.numpy())
    guided_backprop = float_2u8_norm(guided_backprop)
    grad_cam = float_2u8_norm(grad_cam)
    guided_grad_cam = float_2u8_norm(guided_grad_cam)
    integrated_grad = float_2u8_norm(integrated_grad)

    fig, axs = plt.subplots(nrows=3, ncols=2, squeeze=False, figsize=(10, 10))
    
    fig.suptitle(f'{name}\nModel prediction: {pred}\nRetinopathy class label: {r_grade}')

    axs[0, 0].set_title('Original Image')
    axs[0, 0].imshow(image)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Guided Backpropagation')
    axs[0, 1].imshow(guided_backprop)
    axs[0, 1].axis('off')

    axs[1, 0].set_title('GradCam')
    axs[1, 0].imshow(grad_cam, cmap=cmap)
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Guided-GradCam')
    axs[1, 1].imshow(guided_grad_cam, cmap=cmap)
    axs[1, 1].axis('off')

    axs[2, 0].set_title('Integrated Gradinets')
    axs[2, 0].imshow(integrated_grad, cmap=cmap)
    axs[2, 0].axis('off')

    axs[2, 1].set_title('Integrated Gradinets (overlay)')
    axs[2, 1].imshow(integrated_grad, cmap=cmap)
    axs[2, 1].imshow(image, alpha=overlay_alpha)
    axs[2, 1].axis('off')

    plt.tight_layout()
    return fig


@gin.configurable
def visual(model, run_paths, images_folder, image_numbers, train=True,  last_conv_layer=None, binary=True):
    
    logging.info('\n======== Starting DeepVis ========')
    
    labels_path = images_folder + "labels/"
    
    # read data from csv-file
    df_train_val = pd.read_csv(labels_path + "train.csv",
                               usecols=['Image name', 'Retinopathy grade'])
    df_test = pd.read_csv(labels_path + "test.csv",
                          usecols=['Image name', 'Retinopathy grade'])

    suffix_train = 'images/train/'
    suffix_test = 'images/test/'

    path_train = images_folder + suffix_train
    path_test = images_folder + suffix_test

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(
        run_paths['path_ckpts_train']))

    plt.figure(figsize=(10 * 4, 10 * 3))
    i = 0

    # get all images available
    images_list_train = sorted(os.listdir(path_train))
    images_list_test = sorted(os.listdir(path_test))

    # visualize indexes specified (NOTE: Image number starts at 1, therfore we do not count from 0!)
    for index in image_numbers:
        for images_list, path, df in [(images_list_train, path_train, df_train_val), (images_list_test,path_test, df_test)]:
            
            img_array = get_img_array(path + images_list[index-1])
            
            # get retinopathy grade
            r_grade = df.loc[df['Image name'] == images_list[index-1].split('.')[0]]['Retinopathy grade'].values[0]
          
            # get labels and predictions from the model
            pred = tf.argmax(model(img_array), axis=1)
            if pred == 0:
                    pred = 'NRDR'
            elif pred ==1:
                pred = 'RDR'

            # GuidedBackPropagation
            gb = GuidedBackPropagation(model, img_array, last_conv_layer)
            gb_grad_map = gb.get_gb_grad()

            # GradCam
            grad_cam = GradCam(model, img_array, last_conv_layer)
            heatmap = grad_cam.get_heatmap()
            jet_heatmap = grad_cam.get_jet_heatmap(heatmap)
            grad_cam_mixed_img = grad_cam.get_mixed_img(jet_heatmap)

            # Guided_grad_cam
            guided_grad_cam_heatmap = guided_grad_cam(gb_grad_map, jet_heatmap)

            # IntegratedGradients
            integrated_grad = IntegratedGradients(model, img_array)
            integrated_grad_heatmap = integrated_grad.get_heatmap()

            fig = plot_deepvis_outputs(image=tf.squeeze(img_array),
                                    guided_backprop=gb_grad_map,
                                    grad_cam=grad_cam_mixed_img,
                                    guided_grad_cam=guided_grad_cam_heatmap,
                                    integrated_grad=integrated_grad_heatmap,
                                    r_grade=r_grade,
                                    pred=pred,
                                    name=images_list[index-1]
                                    )

            deepvis_path = run_paths["path_plot"] + "/" + path.split('/')[-2] + \
                "/deepvis_" + images_list[index-1]
            logging.info(f"Storing image to {deepvis_path}")
            fig.savefig(deepvis_path)

    logging.info('======== Finished DeepVis ========')
