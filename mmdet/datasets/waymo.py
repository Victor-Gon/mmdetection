# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class WaymoOpenDataset(BaseDetDataset):
    """Dataset for Waymo."""

    METAINFO = {
        'classes':
        ('TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST'),
        # palette is a list of color tuples, which is used for visualization.
        'classwise_iou': {
            'TYPE_VEHICLE': 0.7,
            'TYPE_PEDESTRIAN': 0.5,
            'TYPE_CYCLIST': 0.5
        },
        # class id mapping to "enum Type" in
        # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
        'class_type_to_submit':{
            'TYPE_VEHICLE': 1,
            'TYPE_PEDESTRIAN': 2,
            'TYPE_CYCLIST': 4
        },
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
    }
    
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos




#  def xyxy2cxcywh(self, bbox):
#         _bbox = bbox.tolist()
#         return [
#             (_bbox[0] + _bbox[2]) / 2,
#             (_bbox[1] + _bbox[3]) / 2,
#             _bbox[2] - _bbox[0],
#             _bbox[3] - _bbox[1],
#         ]

#     def _det2dicts(self, results):
#         dict_results = []
#         for idx in range(len(self)):
#             img_info = self.data_infos[idx]
#             result = results[idx]
#             num_valid_labels = min(len(result), len(self.cat_ids))
#             for label in range(num_valid_labels):
#                 bboxes = result[label]
#                 for i in range(bboxes.shape[0]):
#                     cx, cy, w, h = self.xyxy2cxcywh(bboxes[i])
#                     class_name = self.CLASSES[label]
#                     data = dict()
#                     data['filename'] = img_info['filename']
#                     data['context_name'] = img_info['context_name']
#                     data['timestamp_micros'] = img_info['timestamp_micros']
#                     data['camera_name'] = img_info['camera_id']
#                     data['frame_index'] = img_info['frame_id']
#                     data['time_of_day'] = img_info['time_of_day']
#                     data['location'] = img_info['location']
#                     data['weather'] = img_info['weather']
#                     data['center_x'] = cx
#                     data['center_y'] = cy
#                     data['length'] = w  # length: dim x
#                     data['width'] = h  # width: dim y
#                     data['score'] = float(bboxes[i][4])
#                     data['type'] = self.CLASS_TYPE_TO_SUBMIT[class_name]
#                     data['id'] = f'{idx}_{label}_{i}'  # dummy tracking id
#                     dict_results.append(data)

#         return dict_results


# def results2dicts(self, results, outfile_prefix):
#         result_files = dict()
#         if isinstance(results[0], list):
#             dict_results = self._det2dicts(results)
#             result_files['bbox'] = f'{outfile_prefix}.bbox.pkl'
#             mmcv.dump(dict_results, result_files['bbox'])
#         else:
#             raise TypeError('invalid type of results')
#         return result_files

#     def results2proto(self, results, outfile_prefix):
#         if isinstance(results[0], list):
#             dict_results = self._det2dicts(results)

#         detections = metrics_pb2.Objects()

#         for detection in dict_results:
#             obj = metrics_pb2.Object()

#             # fig = plt.figure()
#             # img = mpimg.imread('data/waymococo_f0/val2020/'+detection['filename'])
#             # plt.imshow(img)
#             #
#             # rect = patches.Rectangle((detection['center_x']-detection['length']/2, detection['center_y'] -detection['width']/2 )
#             #                          , detection['length'], detection['width'], linewidth=1, edgecolor='r', facecolor='none')
#             #
#             # ax = plt.gca()
#             # # Add the patch to the Axes
#             # ax.add_patch(rect)
#             # plt.show()

#             lab = label_pb2.Label()
#             lab.box.center_x = detection['center_x']
#             lab.box.center_y = detection['center_y']
#             lab.box.length = detection['length']
#             lab.box.width = detection['width']
#             lab.type = detection['type']

#             obj.object.MergeFrom(lab)
#             if detection['score']:
#                 obj.score = detection['score']

#             obj.context_name = detection["context_name"]
#             obj.frame_timestamp_micros = detection["timestamp_micros"]
#             obj.camera_name = detection["camera_name"]

#             detections.objects.append(obj)

#         f = open(outfile_prefix + ".bin", 'wb')
#         serialized = detections.SerializeToString()
#         f.write(serialized)
#         f.close()

#     def anns2proto(self, outfile_prefix):
#         anns = self.coco.anns
#         ground_truths = metrics_pb2.Objects()

#         for _, ann in anns.items():
#             obj = metrics_pb2.Object()

#             img_info = self.data_infos[ann['image_id']]

#             x1, y1, w, h = ann['bbox']

#             cx = x1 + w / 2
#             cy = y1 + h / 2

#             lab = label_pb2.Label()
#             lab.box.center_x = cx
#             lab.box.center_y = cy
#             lab.box.length = w
#             lab.box.width = h
#             lab.type = 4 if ann['category_id'] == 3 else ann['category_id']
#             lab.detection_difficulty_level = 1 if ann['det_difficult'] == 0 else ann['det_difficult']
#             obj.object.MergeFrom(lab)

#             obj.context_name = img_info["context_name"]
#             obj.frame_timestamp_micros = img_info["timestamp_micros"]
#             obj.camera_name = img_info["camera_id"]

#             ground_truths.objects.append(obj)

#         f = open(outfile_prefix + "_gt.bin", 'wb')
#         serialized = ground_truths.SerializeToString()
#         f.write(serialized)
#         f.close()


    # def format_results(self,
    #                    results,
    #                    outfile_prefix=None,
    #                    format_type='waymo',
    #                    **kwargs):
    #     """Format the results to list[dict] or json.

    #     Args:
    #         results (list[tuple | numpy.ndarray]): Testing results of the
    #             dataset.
    #         outfile_prefix (str | None): The prefix of json files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.

    #     Returns:
    #         tuple: (result_files, tmp_dir), result_files is a dict containing
    #             the json filepaths, tmp_dir is the temporal directory created
    #             for saving json files when outfile_prefix is not specified.
    #     """
    #     assert isinstance(results, list), 'results must be a list'
    #     assert len(results) == len(self), (
    #         'The length of results is not equal to the dataset len: {} != {}'.
    #         format(len(results), len(self)))

    #     if outfile_prefix is None:
    #         tmp_dir = tempfile.TemporaryDirectory()
    #         outfile_prefix = osp.join(tmp_dir.name, 'results')
    #     else:
    #         tmp_dir = None

    #     if format_type == 'waymo':
    #         result_files = self.results2dicts(results, outfile_prefix)
    #         self.results2proto(results, outfile_prefix)
    #         self.anns2proto(outfile_prefix)

    #     elif format_type == 'coco':
    #         result_files = self.results2json(results, outfile_prefix)
    #     else:
    #         raise ValueError('invalid format type')

    #     return result_files, tmp_dir
    
    
    
## EVALUATE ANTIGUO WAYMO    




    