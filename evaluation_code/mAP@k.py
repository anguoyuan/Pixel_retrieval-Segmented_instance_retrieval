# A proper pixel retrieval method should do the pixel-level segmentation and image-level retrieval simultaneously.
# However, there is no such method yet. 
# So we provide the code to calculate the mAP@k of combining pixel-level segmentation and image-level retrieval results.


import numpy as np
from utils.evaluate_pixel import compute_maps
from utils.evaluate import compute_map,ParseEasyMediumHardGroundTruth
from utils.path_manage import name_list
import os

#load the ious result
Result_dir='../results/'

segment_method_list=['example_sift_sp', 'delf_sp']
#segment_method_list=['delg_sp']

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
            if retrieval_dataset=='roxford' or retrieval_dataset=='R1Moxford':
                ious_path=os.path.join(Result_dir,'pixel-level',segment_method,'ious','proxford_ious.npy')
                ious=np.load(ious_path,allow_pickle=True).item()
            elif retrieval_dataset=='rparis' or retrieval_dataset=='R1Mparis':
                ious_path=os.path.join(Result_dir,'pixel-level',segment_method,'ious','prparis_ious.npy')
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
