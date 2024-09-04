import os
import random
import shutil
from preprocess import Parser
from tools import FileTools
from tqdm import tqdm

old_version_path = os.path.abspath('../data/OpenEA_dataset_v1.1/')
new_version_path = os.path.abspath('../data')

os.chdir(old_version_path)
datasets = os.listdir('..')
print(datasets)


def load_oea_file(src, dst, filetype):
    tups = Parser.for_file(src, filetype)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            print(*tup, sep='\t', file=wf)


def load_attr(src, dst):
    tups = Parser.for_file(src, Parser.OEAFileType.ttl_full)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            print(*tup, sep='\t', file=wf)


import pickle

dataset = 'EN_DE_100K_V1'

os.chdir('/'.join((old_version_path, dataset)))

dataset_new_path = '/'.join((new_version_path, dataset))
if not os.path.exists(dataset_new_path):
    os.mkdir(dataset_new_path)
# load_attr('attr_triples_1', '/'.join((dataset_new_path, 'attr_triples_1')))
load_oea_file('attr_triples_1', '/'.join((dataset_new_path, 'attr_triples_1')),
              Parser.OEAFileType.attr)
load_oea_file('attr_triples_2', '/'.join((dataset_new_path, 'attr_triples_2')),
              Parser.OEAFileType.attr)
load_oea_file('rel_triples_1', '/'.join((dataset_new_path, 'rel_triples_1')), Parser.OEAFileType.rel)
# load_oea_file('/'.join((dataset_new_path, 'rel_triples_1')), '/'.join((dataset_new_path, 'rel_triples_1')),
#               Parser.OEAFileType.rel)
load_oea_file('rel_triples_2', '/'.join((dataset_new_path, 'rel_triples_2')), Parser.OEAFileType.rel)
# # load_attr('_'.join((ds2, 'rel_triples')), '/'.join((dataset_new_path, 'rel_triples_2')))
#
shutil.copy('ent_links', '/'.join((dataset_new_path, 'ent_links')))
load_oea_file('/'.join((dataset_new_path, 'ent_links')), '/'.join((dataset_new_path, 'ent_links')),
              Parser.OEAFileType.truth)

ent_links = FileTools.load_list('/'.join((dataset_new_path, 'ent_links')))
random.seed(11037)
random.shuffle(ent_links)
ent_len = len(ent_links)
# train_len = ent_len * 2 // 10
# valid_len = ent_len * 1 // 10
train_len = ent_len * 4 // 15
valid_len = ent_len // 30
train_links = ent_links[:train_len]
valid_links = ent_links[train_len: train_len + valid_len]
test_links = ent_links[train_len + valid_len:]
new_fold_path = '/'.join((new_version_path, dataset, '721_5fold', '1'))
if not os.path.exists(new_fold_path):
    os.makedirs(new_fold_path)
os.chdir(new_fold_path)
FileTools.save_list(train_links, '/'.join((new_fold_path, 'train_links')))
FileTools.save_list(valid_links, '/'.join((new_fold_path, 'valid_links')))
FileTools.save_list(test_links, '/'.join((new_fold_path, 'test_links')))

