# This is the code to calculate the mious of pixel retrieval task.
# The pipeline is: matching results -> ious -> mious
# You should save your matching results in 'Result_dir/method/matching_result', such as './results/sift_sp/matching_result/proxford.npy';
# and then this script will calculate and save the ious and mious in 'Result_dir/method/ious' and 'Result_dir/method/mious'

from gettext import npgettext
import numpy as np
from utils.path_manage import name_list
import os
import json
from utils.iou_compute import extract_bounding_box, extract_segmentation
import cv2
import imageio

USING_EXISTING_IOUS=False #if True, use the existing ious, if False, compute the ious
Result_dir='../results'
GND_folder='../pixel_retrieval_benchmarks/'
image_path = '../revisitop/data/'
visualization_path = '../results/pixel-level/example_sift_sp/visualizations'
print('Saving results to {}'.format(visualization_path))


#change the method list here
method_list=['example_sift_sp']

for dataset in ['proxford','prparis']:
    for method in method_list:
        if not os.path.exists(os.path.join(Result_dir,'pixel-level',method,'visualizations')):
            os.makedirs(os.path.join(Result_dir,'pixel-level',method,'visualizations'))

        matching_file=dataset+'.npy'
        matching_file=os.path.join(Result_dir, 'pixel-level', method, 'matching_result', matching_file)
        match_set=np.load(matching_file, allow_pickle=True).item()

        gnd_seg_dict_path=os.path.join(GND_folder,dataset[2:]+'_pair2labelpath.npy')
        gnd_seg_dict=np.load(gnd_seg_dict_path,allow_pickle=True).item()

        query_list, index_list, gnd = name_list(dataset)  

        # for q in range(70): # Uncomment to visualize results for all query images
        for q in range(1):
            for i in gnd[q]['hard']:
                print('Processing dataset {}: query {} and index {}'.format(dataset, q, i))
                query_img_path = os.path.join(image_path, '{}5k'.format(dataset[2:]), 'jpg', query_list[q]+'.jpg')
                index_img_path = os.path.join(image_path, '{}5k'.format(dataset[2:]), 'jpg', index_list[i]+'.jpg')
                query_img = np.array(cv2.imread(query_img_path, 1)[:, :, ::- 1])
                index_img = np.array(cv2.imread(index_img_path, 1)[:, :, ::- 1])

                found_points=match_set['qimlist_'+str(q)+'_imlist_'+str(i)]

                gnd_seg_path=gnd_seg_dict[str(q)+'_'+str(i)]
                gnd_seg_path = os.path.join(GND_folder, gnd_seg_path)
                gt_points = json.load(open(gnd_seg_path))['shapes'][0]['points']

                directory = os.path.join(visualization_path, f"query_{q}_index_{i}")
                os.makedirs(directory, exist_ok=True)

                query_points = json.load(open(os.path.join(GND_folder, "queries/{}".format(dataset[2:]), "{}.json".format(q))))["shapes"][0]["points"]
                query_points = np.array(query_points, dtype=np.int32)
                query_points = query_points.reshape((-1, 1, 2))
                query_img = cv2.polylines(query_img.copy(), [query_points], isClosed=True, color=(255, 0, 0), thickness=5)

                pred_bbox, pred_bbox_masked = extract_bounding_box(index_img, found_points)
                gt_bbox, gt_bbox_masked = extract_bounding_box(index_img, gt_points)
                pred_seg, pred_seg_masked = extract_segmentation(index_img, found_points)
                gt_seg, gt_seg_masked = extract_segmentation(index_img, gt_points)

                imageio.imwrite(os.path.join(directory, "0_query_img.jpg"), query_img)
                imageio.imwrite(os.path.join(directory, "1_gt_bbox.jpg"), gt_bbox)
                imageio.imwrite(os.path.join(directory, "2_pred_bbox.jpg"), pred_bbox)
                imageio.imwrite(os.path.join(directory, "3_gt_seg.jpg"), gt_seg)
                imageio.imwrite(os.path.join(directory, "4_pred_seg.jpg"), pred_seg)


            for i in gnd[q]['easy']:
                query_img_path = os.path.join(image_path, '{}5k'.format(dataset[2:]), 'jpg', query_list[q]+'.jpg')
                index_img_path = os.path.join(image_path, '{}5k'.format(dataset[2:]), 'jpg', index_list[i]+'.jpg')
                query_img = np.array(cv2.imread(query_img_path))
                index_img = np.array(cv2.imread(index_img_path))

                found_points=match_set['qimlist_'+str(q)+'_imlist_'+str(i)]

                gnd_seg_path=gnd_seg_dict[str(q)+'_'+str(i)]
                gnd_seg_path = os.path.join(GND_folder, gnd_seg_path)
                gt_points = json.load(open(gnd_seg_path))['shapes'][0]['points']

                directory = os.path.join(visualization_path, f"query_{q}_index_{i}")
                os.makedirs(directory, exist_ok=True)

                query_points = json.load(open(os.path.join(GND_folder, "queries/{}".format(dataset[2:]), "{}.json".format(q))))["shapes"][0]["points"]
                query_points = np.array(query_points, dtype=np.int32)
                query_points = query_points.reshape((-1, 1, 2))
                query_img = cv2.polylines(query_img.copy(), [query_points], isClosed=True, color=(255, 0, 0), thickness=5)

                pred_bbox, pred_bbox_masked = extract_bounding_box(index_img, found_points)
                gt_bbox, gt_bbox_masked = extract_bounding_box(index_img, gt_points)
                pred_seg, pred_seg_masked = extract_segmentation(index_img, found_points)
                gt_seg, gt_seg_masked = extract_segmentation(index_img, gt_points)

                imageio.imwrite(os.path.join(directory, "0_query_img.jpg"), query_img)
                imageio.imwrite(os.path.join(directory, "1_gt_bbox.jpg"), gt_bbox)
                imageio.imwrite(os.path.join(directory, "2_pred_bbox.jpg"), pred_bbox)
                imageio.imwrite(os.path.join(directory, "3_gt_seg.jpg"), gt_seg)
                imageio.imwrite(os.path.join(directory, "4_pred_seg.jpg"), pred_seg)
