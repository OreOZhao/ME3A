import os
import random
import shutil
from preprocess import Parser
from tools import FileTools

old_version_path = os.path.abspath('../data/mmkg')
new_version_path = os.path.abspath('../data')

os.chdir(old_version_path)
datasets = os.listdir('.')
print(datasets)


def load_oea_file(src, dst, filetype):
    tups = Parser.for_file(src, filetype)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            print(*tup, sep='\t', file=wf)


def load_ttl_no_compress(src, dst):
    tups = Parser.for_file(src, Parser.OEAFileType.ttl_full)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            print(*tup, sep='\t', file=wf)


def load_attr(src, dst):
    tups = Parser.for_file(src, Parser.OEAFileType.attr)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            print(*tup, sep='\t', file=wf)


def load_rel(src, dst):
    tups = Parser.for_file(src, Parser.OEAFileType.rel)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            print(*tup, sep='\t', file=wf)


def load_links(src, dst):
    tups = Parser.for_file(src, Parser.OEAFileType.ttl_full)
    to_tups = []
    for tup in tups:
        temp = [tup[0], tup[-1]]
        to_tups.append(temp)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in to_tups:
            print(*tup, sep='\t', file=wf)


def load_name_attr(src, dst, ent_name_dict):
    tups = Parser.for_file(src, Parser.OEAFileType.ttl_full)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            lhs = tup[0].strip()
            rel = tup[1].strip()
            rhs = tup[2].strip()
            name_tuple = [ent_name_dict[lhs], rel, rhs]
            print(*name_tuple, sep='\t', file=wf)


def load_name_rel(src, dst, ent_name_dict):
    tups = Parser.for_file(src, Parser.OEAFileType.ttl_full)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            lhs = tup[0].strip()
            rel = tup[1].strip()
            rhs = tup[2].strip()
            name_tuple = [ent_name_dict[lhs], rel, ent_name_dict[rhs]]
            print(*name_tuple, sep='\t', file=wf)


def load_name_links(src, dst, ent_name_dict):
    tups = Parser.for_file(src, Parser.OEAFileType.ttl_full)
    to_tups = []
    for tup in tups:
        temp = [tup[0].strip(), tup[-1].strip()]
        to_tups.append(temp)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in to_tups:
            ent1 = tup[0].strip()
            ent2 = tup[1].strip()
            name_tup = [ent_name_dict[ent1], ent2]
            print(*name_tup, sep='\t', file=wf)


datasets = ['DB15K', 'FB15K', 'YAGO15K']
file_lists = ['_EntityTriples.txt', '_ImageIndex.txt', '_NumericalTriples.txt', '_SameAsLink.txt', '_EntityDscp.txt']
to_file_lists = ['attr_triples_1', 'attr_triples_2', 'rel_triples_1', 'rel_triples_2', 'ent_links']




dataset = 'FB15K'
os.chdir('/'.join((old_version_path, dataset)))
ent_name_file = 'ent_name.txt'

ent_name = FileTools.load_dict(ent_name_file)
attr_file = dataset + '_NumericalTriples.txt'
to_attr_file = 'name_attr_triples'
load_name_attr(attr_file, to_attr_file, ent_name)
rel_file = dataset + '_EntityTriples.txt'
to_rel_file = 'name_rel_triples'
load_name_rel(rel_file, to_rel_file, ent_name)

dataset = 'DB15K'
os.chdir('/'.join((old_version_path, dataset)))
links_file = dataset + '_SameAsLink.txt'
to_links_file = 'name_ent_links'
load_name_links(links_file, to_links_file, ent_name)

dataset = 'YAGO15K'
os.chdir('/'.join((old_version_path, dataset)))
links_file = dataset + '_SameAsLink.txt'
to_links_file = 'name_ent_links'
load_name_links(links_file, to_links_file, ent_name)


dataset = 'fb_yg_15k'
os.chdir('/'.join((old_version_path, dataset)))
rel_file = 'rel_triples_1'
load_oea_file(rel_file, rel_file, Parser.OEAFileType.ttl_full)
load_oea_file(rel_file, rel_file, Parser.OEAFileType.rel)
attr_file = 'attr_triples_1'
load_oea_file(attr_file, attr_file, Parser.OEAFileType.ttl_full)
load_oea_file(attr_file, attr_file, Parser.OEAFileType.attr)
link_file = 'ent_links'
load_oea_file(link_file, link_file, Parser.OEAFileType.truth)

dataset_new_path = '/'.join((new_version_path, dataset))
if not os.path.exists(dataset_new_path):
    os.mkdir(dataset_new_path)
shutil.copy('attr_triples_1', dataset_new_path)
shutil.copy('attr_triples_2', dataset_new_path)
shutil.copy('ent_links', dataset_new_path)
shutil.copy('rel_triples_1', dataset_new_path + '/rel_triples_1')
shutil.copy('rel_triples_2', dataset_new_path + '/rel_triples_2')


ent_links = FileTools.load_list('ent_links')
random.seed(11037)
random.shuffle(ent_links)
ent_len = len(ent_links)
train_len = ent_len * 19 // 100
valid_len = ent_len * 1 // 100
train_links = ent_links[:train_len]
valid_links = ent_links[train_len: train_len + valid_len]
test_links = ent_links[train_len + valid_len:]
new_fold_path = '/'.join((new_version_path, dataset, '721_5fold', '2'))
if not os.path.exists(new_fold_path):
    os.makedirs(new_fold_path)
os.chdir(new_fold_path)

FileTools.save_list(train_links, '/'.join((new_fold_path, 'train_links')))
FileTools.save_list(valid_links, '/'.join((new_fold_path, 'valid_links')))
FileTools.save_list(test_links, '/'.join((new_fold_path, 'test_links')))
