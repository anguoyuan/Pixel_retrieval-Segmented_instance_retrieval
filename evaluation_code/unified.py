# this is for evaluate a method which gives different matching results for different ranking list, like the hypergraph propagation method.

from gettext import npgettext
import numpy as np
from utils.path_manage import name_list
import os
import json
from utils.iou_compute import compute_iou



USING_EXISTING_IOUS=False #if True, use the existing ious, if False, compute the ious
Result_dir='../results/'
GND_folder='../pixel_retrieval_benchmarks/'


#change the method list here
method_list=['hp']

for dataset in ['pR1Moxford','pR1Mparis']:#['proxford','prparis','pR1Moxford','pR1Mparis']:
    for method in method_list:
        if not os.path.exists(os.path.join(Result_dir,'pixel-level',method,'ious')):
            os.makedirs(os.path.join(Result_dir,'pixel-level',method,'ious'))
        if not os.path.exists(os.path.join(Result_dir,'pixel-level',method,'mious')):
            os.makedirs(os.path.join(Result_dir,'pixel-level',method,'mious'))

        if USING_EXISTING_IOUS:
            IOU_path=os.path.join(Result_dir,'pixel-level',method,'ious', dataset+'_ious.npy')
            IOU_dict=np.load(IOU_path,allow_pickle=True).item()
        else:
            IOU_dict={}

            matching_file=dataset+'.npy'
            matching_file=os.path.join(Result_dir,'pixel-level',method, 'matching_result' ,matching_file)
            match_set=np.load(matching_file, allow_pickle=True).item()

            if 'oxford' in dataset:
                gnd_seg_dict_path=os.path.join(GND_folder,'oxford_pair2labelpath.npy')
                gnd_seg_dict=np.load(gnd_seg_dict_path,allow_pickle=True).item()
            elif 'paris' in dataset:
                gnd_seg_dict_path=os.path.join(GND_folder,'paris_pair2labelpath.npy')
                gnd_seg_dict=np.load(gnd_seg_dict_path,allow_pickle=True).item()

        query_list, index_list,gnd=name_list(dataset)  

        medium_bbox_mious=[]
        medium_seg_mious=[]
        hard_bbox_mious=[]
        hard_seg_mious=[]
        for q in range(70):
            q_medium_bbox_ious=[]
            q_medium_seg_ious=[]
            q_hard_bbox_ious=[]
            q_hard_seg_ious=[]
            for i in gnd[q]['hard']:

                if USING_EXISTING_IOUS:
                    bbox_iou,segmentation_iou=IOU_dict[str(q)+'_'+str(i)]
                else:
                    try:
                        found_points=match_set['qimlist_'+str(q)+'_imlist_'+str(i)]
                    except:
                        #bbox_seg_ious[str(q)+'_'+str(i)]=[0,0]
                        bbox_iou=0
                        segmentation_iou=0
                        IOU_dict[str(q)+'_'+str(i)]=[bbox_iou,segmentation_iou]

                        q_medium_bbox_ious.append(bbox_iou)
                        q_medium_seg_ious.append(segmentation_iou)
                        q_hard_bbox_ious.append(bbox_iou)
                        q_hard_seg_ious.append(segmentation_iou)
                        continue
                    if len(found_points) == 0:
                        #bbox_seg_ious[str(q)+'_'+str(i)]=[0,0]
                        bbox_iou=0
                        segmentation_iou=0
                        IOU_dict[str(q)+'_'+str(i)]=[bbox_iou,segmentation_iou]

                        q_medium_bbox_ious.append(bbox_iou)
                        q_medium_seg_ious.append(segmentation_iou)
                        q_hard_bbox_ious.append(bbox_iou)
                        q_hard_seg_ious.append(segmentation_iou)
                        continue

                    gnd_seg_path=gnd_seg_dict[str(q)+'_'+str(i)]
                    gnd_seg_path = os.path.join(GND_folder, gnd_seg_path)
                    gt_label = json.load(open(gnd_seg_path))
                    #gt_points = gt_label['shapes'][0]['points']
                    gt_points = []
                    for shape in gt_label['shapes']:
                        gt_points.append(shape['points'])

                    try:
                        bbox_iou= compute_iou(found_points, gt_points, type='bbox')
                        segmentation_iou= compute_iou(found_points, gt_points,  type='segmentation')
                        IOU_dict[str(q)+'_'+str(i)]=[bbox_iou,segmentation_iou]
                    except:
                        print('error pair: ', str(q)+'_'+str(i))

                q_medium_bbox_ious.append(bbox_iou)
                q_medium_seg_ious.append(segmentation_iou)
                q_hard_bbox_ious.append(bbox_iou)
                q_hard_seg_ious.append(segmentation_iou)

            
            for i in gnd[q]['easy']:
                if USING_EXISTING_IOUS:
                    bbox_iou,segmentation_iou=IOU_dict[str(q)+'_'+str(i)]
                else:
                    try:
                        found_points=match_set['qimlist_'+str(q)+'_imlist_'+str(i)]
                    except:
                        #bbox_seg_ious[str(q)+'_'+str(i)]=[0,0]
                        bbox_iou=0
                        segmentation_iou=0
                        continue
                    if len(found_points) == 0:
                        #bbox_seg_ious[str(q)+'_'+str(i)]=[0,0]
                        bbox_iou=0
                        segmentation_iou=0
                        continue

                    gnd_seg_path=gnd_seg_dict[str(q)+'_'+str(i)]
                    gnd_seg_path = os.path.join(GND_folder, gnd_seg_path)
                    gt_label = json.load(open(gnd_seg_path))
                    #gt_points = gt_label['shapes'][0]['points']
                    gt_points = []
                    for shape in gt_label['shapes']:
                        gt_points.append(shape['points'])

                    try:
                        bbox_iou= compute_iou(found_points, gt_points, type='bbox')
                        segmentation_iou= compute_iou(found_points, gt_points,  type='segmentation')
                        IOU_dict[str(q)+'_'+str(i)]=[bbox_iou,segmentation_iou]
                    except:
                        print('error pair: ', str(q)+'_'+str(i))

                q_medium_bbox_ious.append(bbox_iou)
                q_medium_seg_ious.append(segmentation_iou)

            medium_bbox_mious.append(np.mean(q_medium_bbox_ious))
            medium_seg_mious.append(np.mean(q_medium_seg_ious))
            hard_bbox_mious.append(np.mean(q_hard_bbox_ious))
            hard_seg_mious.append(np.mean(q_hard_seg_ious))
            print(dataset,'query ', q, 'results are: ',np.mean(q_medium_bbox_ious),np.mean(q_medium_seg_ious), np.mean(q_hard_bbox_ious),np.mean(q_hard_seg_ious))

        medium_bbox_mmiou=np.mean(medium_bbox_mious)
        medium_seg_mmiou=np.mean(medium_seg_mious)
        hard_bbox_mmiou=np.mean(hard_bbox_mious)
        hard_seg_mmiou=np.mean(hard_seg_mious)
        print(method, dataset)
        print('medium_bbox_mmiou is: ',medium_bbox_mmiou )
        print('medium_seg_mmiou is: ',medium_seg_mmiou )
        print('hard_bbox_mmiou is: ',hard_bbox_mmiou )
        print('hard_seg_mmiou is: ',hard_seg_mmiou )

        result={'mmiou':[medium_bbox_mmiou,medium_seg_mmiou,hard_bbox_mmiou,hard_seg_mmiou],
        'mious':[medium_bbox_mious,medium_seg_mious,hard_bbox_mious,hard_seg_mious],
        'explain':'mmiou and mious in the order of [medium_bbox_,medium_seg_,hard_bbox_,hard_seg_]'
        }

        #save the result
        np.save(os.path.join(Result_dir,'pixel-level',method,'mious',dataset+'.npy'), result)
        #save ious
        if not USING_EXISTING_IOUS:
            np.save(os.path.join(Result_dir,'pixel-level',method,'ious',dataset+'_ious.npy'), IOU_dict)


# calculate the mAP@k of pixel-level retrieval results.
print('calculating the mAP@k of pixel-level retrieval results.')
import numpy as np
from utils.evaluate_pixel import compute_maps
from utils.evaluate import compute_map,ParseEasyMediumHardGroundTruth
from utils.path_manage import name_list
import os

#load the ious result
Result_dir='../results/'

segment_method_list=['hp']

for retrieval_dataset in ['roxford', 'rparis', 'R1Moxford','R1Mparis']:

    #load the ground truth file
    if retrieval_dataset=='roxford' or retrieval_dataset=='R1Moxford':
        query_list, index_list, ground_truth = name_list(retrieval_dataset)
    elif retrieval_dataset=='rparis' or retrieval_dataset=='R1Mparis':
        query_list, index_list, ground_truth = name_list(retrieval_dataset)
    (_, medium_ground_truth,hard_ground_truth) = ParseEasyMediumHardGroundTruth(ground_truth) # 'ok' and 'junk'

    for imrank_method in ['hp']:
        #load the img rank list
        if retrieval_dataset=='roxford':
            qranks=np.load(os.path.join(Result_dir,'image-level',imrank_method,'roxford.npy'))
        elif retrieval_dataset=='rparis':
            qranks=np.load(os.path.join(Result_dir,'image-level',imrank_method,'rparis.npy'))    
        elif retrieval_dataset=='R1Moxford':
            qranks=np.load(os.path.join(Result_dir,'image-level',imrank_method,'R1Moxford.npy'))    
        elif retrieval_dataset=='R1Mparis':
            qranks=np.load(os.path.join(Result_dir,'image-level',imrank_method,'R1Mparis.npy'))


        for segment_method in segment_method_list:
            ious_path=os.path.join(Result_dir,'pixel-level',segment_method,'ious','p'+retrieval_dataset+'_ious.npy')
            ious=np.load(ious_path,allow_pickle=True).item()

            total_ranks_=qranks.T
            im_medium_metrics = compute_map(total_ranks_.T,medium_ground_truth) 
            im_hard_metrics=compute_map(total_ranks_.T, hard_ground_truth)

            p_medium_metrics = compute_maps(retrieval_dataset, total_ranks_.T,ious, medium_ground_truth, (50,100,5)) 
            p_hard_metrics=compute_maps(retrieval_dataset, total_ranks_.T, ious, hard_ground_truth,(50,100,5))

            print(retrieval_dataset, imrank_method, segment_method)
            print('medium image level mAP is ',im_medium_metrics[0])
            print('hard image level mAP is ',im_hard_metrics[0])
            print('medium pixel level mAP is ', p_medium_metrics)
            print('hard pixel level mAP is ', p_hard_metrics)
