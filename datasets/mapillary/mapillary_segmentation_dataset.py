import os

import numpy as np
import chainer
from chainercv.utils import read_image
from datasets.mapillary import mapillary_utils


class MapillarySegmentationDataset(chainer.dataset.DatasetMixin):

    """Dataset class for the semantic segmantion task of mapillary.

    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`chainercv.datasets.voc_semantic_segmentation_label_names`.

    .. _`VOC2012`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.

    """

    def __init__(self, data_dir='auto', split='train'):
        if split not in ['train', 'trainval', 'val']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')

        if data_dir == 'auto':
            data_dir = mapillary_utils.get_mapillary()

        id_list_file = os.path.join(data_dir, 'training')

        # foreach files
        self.ids = []
        files = os.listdir(id_list_file)
        for id_ in files:
            path = os.path.join(id_list_file, id_)
            ext_name = os.path.splitext(path)[1]
            if ext_name == 'jpg':
                file_name = id_.split('.')
                id_ = file_name[0]
                self.ids.append(id_)

        self.data_dir = data_dir

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and a label image. The color image is in CHW
        format and the label image is in HW format.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of color image and label whose shapes are (3, H, W) and
            (H, W) respectively. H and W are height and width of the
            images. The dtype of the color image is :obj:`numpy.float32` and
            the dtype of the label image is :obj:`numpy.int32`.

        """
        if i >= len(self):
            raise IndexError('index is too large')
        img_file = os.path.join(
            self.data_dir, 'training', self.ids[i] + '.jpg')
        img = read_image(img_file, color=True)
        label = self._load_label(self.data_dir, self.ids[i])
        return img, label

    @staticmethod
    def _load_label(data_dir, id_):
        label_file = os.path.join(
            data_dir, 'labels', id_ + '.png')
        label = read_image(label_file, dtype=np.int32, color=False)
        label[label == 255] = -1
        # (1, H, W) -> (H, W)
        return label[0]
