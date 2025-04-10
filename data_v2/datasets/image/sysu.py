from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re
import warnings

from .. import ImageDataset


class SYSU(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'SYSU'


    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(SYSU, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))  # 파일 형식은 .jpg로 설정
        # [id]_[camera_id]_[frame_id].jpg 패턴에 맞는 정규식
        pattern = re.compile(r'([0-9]+)_([0-9]+)_[0-9]+\.jpg')  # 예: 0002_3_0001.jpg

        pid_container = set()
        for img_path in img_paths:
            # 파일명에서 PID와 카메라 ID를 추출
            match = pattern.search(img_path)
            if match:
                pid = int(match.group(1))  # ID: 앞의 숫자 (0002)
                camid = int(match.group(2))  # Camera ID: 가운데 숫자 (3)
                pid_container.add(pid)
            else:
                continue  # 패턴에 맞지 않으면 skip

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        data = []
        for img_path in img_paths:
            match = pattern.search(img_path)
            if match:
                pid = int(match.group(1))  # ID
                camid = int(match.group(2)) - 1  # Camera ID (0-based index)
                if relabel:
                    pid = pid2label[pid]
                data.append((img_path, pid, camid))

        return data