import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            return layer.name


class GuidedBackPropagation:
    def __init__(self, model, image_array, last_conv_layer=None):
        self.model = model
        self.last_conv_layer = last_conv_layer
        self.inputs =  tf.convert_to_tensor(image_array)

    @tf.custom_gradient
    def guided_relu(self, x):
        y = tf.nn.relu(x)
        def grad(dy):
            return tf.cast(x > 0, tf.float32) * tf.cast(dy > 0, tf.float32) * dy
        return y, grad

    def get_gb_grad(self):
        if not self.last_conv_layer:
            self.last_conv_layer = get_last_conv_layer(self.model)

        gb_model = keras.Model(
            self.model.inputs, self.model.get_layer(self.last_conv_layer).output
        )
        layers_list = [layer for layer in self.model.layers[1:] if hasattr(layer, "activation")]
        for layer in layers_list:
            if layer.activation == keras.activations.relu:
                layer.activation = self.guided_relu

        with tf.GradientTape() as tape:
            tape.watch(self.inputs)
            out = gb_model(self.inputs)

        gradient = tape.gradient(out, self.inputs)[0]
        # shape of gradient (256,256,3)
        gradient = self.process_grad(gradient)
        return gradient

    def process_grad(self, grad):
        grad = (grad - tf.reduce_mean(grad)) / (tf.math.reduce_std(grad) + 1e-5)
        grad *= 0.2
        grad += 0.5
        grad = tf.clip_by_value(grad, 0, 1)
        return grad.numpy()



class GradCam:
    def __init__(self, model, image_array, last_conv_layer=None):
        self.model = model
        self.last_conv_layer = last_conv_layer
        self.img_array = image_array

    def get_heatmap(self):
        if self.last_conv_layer:
            pass
        else:
            self.last_conv_layer = get_last_conv_layer(self.model)

        # input one image. shape=(1, 256, 256, 3)  >>> last_conv_layer_output.shape e.g. (1, 10, 10, 16)
        grad_model = keras.Model(
            self.model.inputs, [self.model.get_layer(self.last_conv_layer).output, self.model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, logits = grad_model(self.img_array)
            index = tf.argmax(tf.squeeze(logits))
            pred = logits[:, index]

        gradients = tape.gradient(pred, last_conv_layer_output)
        # gradients.shape (1, 10, 10, 16)
        mean_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        mean_gradients = tf.reshape(mean_gradients, (-1, 1))

        weighted_map = tf.squeeze(
            tf.matmul(last_conv_layer_output, mean_gradients))
        # add relu to this weighted map and normalize. resize to the shape of image later
        heatmap = tf.maximum(weighted_map, 0) / tf.reduce_max(weighted_map)

        return heatmap.numpy()

    def get_jet_heatmap(self, heatmap):
        heatmap = np.uint8(255 * heatmap)
        jet = plt.cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((self.img_array.shape[1], self.img_array.shape[2]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        # shape of jet_heatmap (256, 256, 3)
        return  jet_heatmap / jet_heatmap.max()

    def get_mixed_img(self, jet_heatmap,  factor=0.5):
        img_array = tf.squeeze(self.img_array)
        # rescale image to have same range as jet_heatmap
        img_array = img_array / img_array.numpy().max()
        mixed_img = jet_heatmap * factor + img_array
        mixed_img = mixed_img / mixed_img.numpy().max()
        return mixed_img.numpy()



def guided_grad_cam(grad_map, jet_heatmap):
    guided_grad_cam_heatmap = grad_map * jet_heatmap
    return guided_grad_cam_heatmap

class IntegratedGradients:
    """
    Implmetation of integrted gradients
    
    This implementation follows the tf tutorial from 
    https://www.tensorflow.org/tutorials/interpretability/integrated_gradients?hl=en
    """
    
    def __init__(self, model, image_array, last_conv_layer=None):
        self.model = model
        self.last_conv_layer = last_conv_layer
        self.inputs =  tf.convert_to_tensor(tf.squeeze(image_array))
        
        # black grayscale image as baseline
        self.baseline_img = tf.zeros(shape=self.inputs.shape, dtype=self.inputs.dtype) 
         
        # generate m_steps intervals for integral approximation
        self.m_steps = 50  
        self.alphas = tf.linspace(start=0.0,stop=1.0,num=self.m_steps+1) 
    
    def get_heatmap(self):

        interpolated_images = self.interpolate_images()

        # compute gradients w.r.t. all pixels of all interpolated images
        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            preds = self.model(interpolated_images)
        path_grads = tape.gradient(preds, interpolated_images)
        
        avg_grads = tf.math.reduce_mean(path_grads, axis=0)
        integrated_grads = (self.inputs - self.baseline_img) * avg_grads
    
        # visualize int-grads by taking absolute value and summing across color channels,
        # as suggested in paper "SmoothGrad: removing noise by adding noise" and
        # "https://www.tensorflow.org/tutorials/interpretability/integrated_gradients"
        
        # rgb_img = cv.cvtColor(self.inputs, cv.COLOR_BGR2RGB)
        attribution_mask = tf.reduce_sum(tf.math.abs(integrated_grads), axis=-1)
        
        # norm heatmap 
        heatmap = tf.cast(((attribution_mask-tf.reduce_min(attribution_mask)) \
                    / (tf.reduce_max(attribution_mask)-tf.reduce_min(attribution_mask))) *255.0, tf.uint8)

        return cv.applyColorMap(heatmap.numpy(), cv.COLORMAP_INFERNO)

    def interpolate_images(self):
        """ interpolates linearly between baseline and image """

        alphas_x = self.alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(self.baseline_img, axis=0)
        input_x = tf.expand_dims(self.inputs, axis=0)
        delta = input_x - baseline_x
        images = baseline_x +  alphas_x * delta
        return images
