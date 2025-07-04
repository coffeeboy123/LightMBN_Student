from __future__ import absolute_import
from __future__ import print_function

from .dataset import Dataset, ImageDataset, VideoDataset
from .image import *
from .video import *


__image_datasets = {
    "regdb": RegDB,
    "sysu": SYSU,
    "cuhk03": CUHK03,
    "cuhk03_detected": CUHK03_Detected,
    "cuhk03_labeled": CUHK03_Labeled,
    "cuhk03_splited": CUHK03_splited,
    "dukemtmc": DukeMTMCreID,
    "msmt17": MSMT17,
    "viper": VIPeR,
    "grid": GRID,
    "cuhk01": CUHK01,
    "ilids": iLIDS,
    "sensereid": SenseReID,
    "prid": PRID,
    "cuhk02": CUHK02,
    "mot17": MOT17,
    "aicity24": AICity24Balanced,
    "market1501": Market1501,
}


__video_datasets = {
    "mars": Mars,
    "ilidsvid": iLIDSVID,
    "prid2011": PRID2011,
    "dukemtmcvidreid": DukeMTMCVidReID,
}


def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            "but expected to be one of {}".format(name, avai_datasets)
        )
    return __image_datasets[name](**kwargs)


def init_video_dataset(name, **kwargs):
    """Initializes a video dataset."""
    avai_datasets = list(__video_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            "but expected to be one of {}".format(name, avai_datasets)
        )
    return __video_datasets[name](**kwargs)


def register_image_dataset(name, dataset):
    """Registers a new image dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::

        import torchreid
        import NewDataset
        torchreid.data.register_image_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources=['new_dataset', 'dukemtmcreid']
        )
    """
    global __image_datasets
    curr_datasets = list(__image_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            "The given name already exists, please choose "
            "another name excluding {}".format(curr_datasets)
        )
    __image_datasets[name] = dataset


def register_video_dataset(name, dataset):
    """Registers a new video dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::

        import torchreid
        import NewDataset
        torchreid.data.register_video_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources=['new_dataset', 'ilidsvid']
        )
    """
    global __video_datasets
    curr_datasets = list(__video_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            "The given name already exists, please choose "
            "another name excluding {}".format(curr_datasets)
        )
    __video_datasets[name] = dataset
