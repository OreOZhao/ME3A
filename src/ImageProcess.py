import os
import random
import shutil
from preprocess import Parser
from tools import FileTools
from tqdm import tqdm
import pickle
import numpy as np

entities_out1 = '../outputs/ME3A-relation/ja_en/entities_tab_1.txt'
entities_out2 = '../outputs/ME3A-relation/ja_en/entities_tab_2.txt'
entity_list1 = FileTools.load_list(entities_out1)
entity_ids1 = {ent: int(s_eid) for s_eid, ent in entity_list1}
entities1 = [ent for s_eid, ent in entity_list1]
entity_list2 = FileTools.load_list(entities_out2)
entity_ids2 = {ent: int(s_eid) for s_eid, ent in entity_list2}
entities2 = [ent for s_eid, ent in entity_list2]

eva_ents1 = '../data/pkls/ja_en_ent/ent_ids_1'
eva_ents2 = '../data/pkls/ja_en_ent/ent_ids_2'
eva_ent_ids1 = FileTools.load_list(eva_ents1)
eva_ent_ids2 = FileTools.load_list(eva_ents2)
eva_tups1 = Parser.for_file(eva_ents1, Parser.OEAFileType.truth)
eva_tups2 = Parser.for_file(eva_ents2, Parser.OEAFileType.truth)
eva_ents = {}
for t in eva_tups1:
    eva_ents[int(t[0])] = t[1]
for t in eva_tups2:
    eva_ents[int(t[0])] = t[1]

eva_ents2id = {v: k for k, v in eva_ents.items()}

eva_img_file = '../data/pkls/ja_en_GA_id_img_feature_dict.pkl'
eva_img_feat = pickle.load(open(eva_img_file, 'rb'))

img_out1 = '../outputs/ME3A-relation/ja_en/img_feat_pickle_1.pickle'
img_out2 = '../outputs/ME3A-relation/ja_en/img_feat_pickle_2.pickle'

img_dict1 = {}
for i, ent in enumerate(entities1):
    if eva_ents2id[ent] in eva_img_feat:
        img_dict1[i] = eva_img_feat[eva_ents2id[ent]]
    else:
        img_dict1[i] = np.random.normal(size=(2048,))
img_dict2 = {}
for i, ent in enumerate(entities2):
    if eva_ents2id[ent] in eva_img_feat:
        img_dict2[i] = eva_img_feat[eva_ents2id[ent]]
    else:
        img_dict2[i] = np.random.normal(size=(2048,))

pickle.dump(img_dict1, open(img_out1, 'wb'))
pickle.dump(img_dict2, open(img_out2, 'wb'))
