import matplotlib
matplotlib.use('Agg')

import os

# from io import StringIO
import io as StringIO

import chainer
import matplotlib.pyplot as plot
import numpy as np
from chainer import serializers
from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.utils import read_image
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_label
from skimage import io

from datasets import cityscapes_label_colors
from datasets import cityscapes_label_names
from datasets import cityscapes_labels
from evaluate import inference
from evaluate import preprocess

from pspnet import PSPNet
from PIL import Image

class ModelPSPNet:
    def __init__(self, model):
        chainer.config.stride_rate = float(2.0 / 3.0)
        chainer.config.save_test_image = False
        chainer.config.train = False
        self.scales = None
        self.gpu = 0
        self.model = model
        self.img_fn = 'test.jpg'

        self.n_class = 19
        self.n_blocks = [3, 4, 23, 3]
        self.feat_size = 90
        self.mid_stride = True
        self.param_fn = 'weights/pspnet101_cityscapes_713_reference.chainer'
        self.base_size = 2048
        self.crop_size = 713
        self.labels = cityscapes_label_names
        self.colors = cityscapes_label_colors

        if self.model == 'VOC':
            self.n_class = 21
            self.n_blocks = [3, 4, 23, 3]
            self.feat_size = 60
            self.mid_stride = True
            self.param_fn = 'weights/pspnet101_VOC2012_473_reference.chainer'
            self.base_size = 512
            self.crop_size = 473
            self.labels = voc_semantic_segmentation_label_names
            self.colors = voc_semantic_segmentation_label_colors
        elif self.model == 'Cityscapes':
            self.n_class = 19
            self.n_blocks = [3, 4, 23, 3]
            self.feat_size = 90
            self.mid_stride = True
            self.param_fn = 'weights/pspnet101_cityscapes_713_reference.chainer'
            self.base_size = 2048
            self.crop_size = 713
            self.labels = cityscapes_label_names
            self.colors = cityscapes_label_colors
        elif self.model == 'ADE20K':
            self.n_class = 150
            self.n_blocks = [3, 4, 6, 3]
            self.feat_size = 60
            self.mid_stride = False
            self.param_fn = 'weights/pspnet101_ADE20K_473_reference.chainer'
            self.base_size = 512
            self.crop_size = 473

    def load_image(self, image_data):
        f = StringIO.BytesIO(image_data)
        image = Image.open(f)
        if image.mode not in ('L', 'RGB'):
            image = image.convert('RGB')

        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            return image[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W) and
            return image.transpose(2, 0, 1)

    def do(self, image_data):
        model = PSPNet(self.n_class, self.n_blocks, self.feat_size, mid_stride=self.mid_stride)
        serializers.load_npz(self.param_fn, model)
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            model.to_gpu(self.gpu)

        img = preprocess(self.load_image(image_data))

        # Inference
        pred = inference(
            model, self.n_class, self.base_size, self.crop_size, img, self.scales)

        # Save the result image
        # ax = vis_image(img)
        # _, legend_handles = vis_label(pred, self.labels, self.colors, alpha=1.0, ax=ax)
        # ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=2,
        #           borderaxespad=0.)
        # base = os.path.splitext(os.path.basename(self.img_fn))[0]
        # plot.savefig('predict_{}.png'.format(base), bbox_inches='tight', dpi=400)

        # if self.model == 'Cityscapes':
        #     label_out = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
        #     for label in cityscapes_labels:
        #         label_out[np.where(pred == label.trainId)] = label.color
        #
        #     io.imsave(
        #         'predict_{}_color({}).png'.format("ss", self.scales), label_out)
        label_out = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
        for label in cityscapes_labels:
            label_out[np.where(pred == label.trainId)] = label.color

        pred_data = StringIO.BytesIO()
        pred_img = Image.fromarray(label_out)
        pred_img.save(pred_data, format="PNG")
        pred_data = pred_data.getvalue()

        print('finish')
        return pred_data
