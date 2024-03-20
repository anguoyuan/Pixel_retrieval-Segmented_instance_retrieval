# -*- coding: utf-8 -*-
"""
prepare the index_list, gnd, and img_path
Created on Tue Dec 28 14:47:50 2021

@author: Guoyuan An
"""


def name_list(db='roxford'):
    #db='rparis'
    #db='roxford'
    #db='R1Moxford'
    env='server'
    global environment, dataset,rop_dir
    environment=env
    
    # set the path of the dataset for different environments
    if env=='server':
        rop_dir='../pixel_retrieval_benchmarks/revisitop/'
    elif env=='pc':
        rop_dir='../pixel_retrieval_benchmarks/revisitop/'
    
    dataset=db
    
    query_list,index_list,ground_truth=_name_list(db)
    print(len(index_list))
    
    return query_list, index_list,ground_truth

#prepare_path(env='pc', db='roxford')
#query_list, index_list,ground_truth=name_list()
############################################################################
def _name_list(retrieval_dataset):
    print('loading name list')
    import pickle
    
    global query_list, index_list, pkl
    
    if 'oxford' in retrieval_dataset:
        pkl_path=rop_dir+'data/roxford5k/gnd_roxford5k.pkl'
    elif 'paris' in retrieval_dataset:
        pkl_path=rop_dir+'data/rparis6k/gnd_rparis6k.pkl'
    
    with open(pkl_path, 'rb') as f:
        pkl=pickle.load(f)
    query_list,index_list, ground_truth=pkl['qimlist'], pkl['imlist'],pkl['gnd']
    
    if retrieval_dataset=='R1Moxford' or retrieval_dataset=='R1Mparis':
        with open(rop_dir+'data/revisitop1m.txt','r') as f:
            distractor_list=f.readlines()
        index_list=index_list+distractor_list
        
    print('loading name list finished')
    return query_list, index_list,ground_truth
