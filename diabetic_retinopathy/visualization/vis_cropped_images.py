import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import cv2

CLASS_NAME = ["NRDR", "RDR"]

images_path = "C:\\Users\\peter\\OneDrive\\Uni\\Master\\4. Semester\\01 Deep Learning Lab\\dl-lab-21w-team20\\diabetic_retinopathy\\data\\IDRID_dataset\\images\\train\\"
labels_path = "C:\\Users\\peter\\OneDrive\\Uni\\Master\\4. Semester\\01 Deep Learning Lab\\dl-lab-21w-team20\\diabetic_retinopathy\\data\\IDRID_dataset\\labels\\train.csv"


def preprocess_image_peter(image):
    """
    Take original image and crop it as good as possible from left & right.
    Afterwards resize it to square (3400x3400 -> often results had about that size)

    Args:
        image (np.array): input non-square image with many black pixels at sides

    Returns:
        np.array : square image (3400x3400) with less black pixels at sides
    """
    # grayscale image for easier thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binary threshold
    _, threshold = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    # get nonzero positions (left & right)
    pos = np.nonzero(threshold)
    right_boundary = pos[1].max()
    left_boundary = pos[1].min()
    # crop image where possible (left & right)
    image = image[:, left_boundary:right_boundary]
    # computations to obtain square image (padding at desired positions)
    upper_diff = (image.shape[1] - image.shape[0]) // 2
    lower_diff = image.shape[1] - image.shape[0] - upper_diff
    image = cv2.copyMakeBorder(image, upper_diff, lower_diff, 0, 0, cv2.BORDER_CONSTANT)

    # convert from BGR to RGB
    # NOTE: open-cv uses BGR as standart, while RGB is most commonly used
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return cv2.resize(image_rgb, (3400, 3400))


def show_image(image_list):

    plt.figure(figsize=(20, 20))
    for n, (image, label) in enumerate(image_list):
        ax = plt.subplot(5, 5, n + 1)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.title(label)
        plt.axis("off")
        if n > 25:
            break
    plt.show()
    
    
def vis_number_of_images(images):
    'Helper class for dataset visulaization'
    plt.figure(0)
    lst = list(range(1,25+1))
    for i in range(5):
        for j in range(4):
            plt.subplot2grid((5,4), (i,j))
            plt.imshow(images[lst[0]])
            lst.pop(0)
    plt.show()
    
def vis_image_without_axis(image):
    'Helper class for paper illustrations'
    from matplotlib import pyplot as plt
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(im_rgb, interpolation='nearest')
    plt.show()


if __name__ == '__main__':

    df = pd.read_csv(labels_path, usecols=['Image name', 'Retinopathy grade'])
    
    i = 0
    image_list = []
    for _, row in df.iterrows():
        image = cv2.imread(images_path + row["Image name"] + ".jpg")
        label = 0 if row['Retinopathy grade'] >= 1 else 1
        preprocessed_image = preprocess_image_peter(image)
        image_list.append((preprocessed_image, label))
        i += 1
        if i >= 25:
            break

    show_image(image_list)

    print('done')
