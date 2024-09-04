import argparse
import datetime
import os
import random
import sys
import torch as t
import numpy as np
from transformers import BertConfig, BertTokenizer

from tools import Logger

seed = 11037
random.seed(seed)
t.manual_seed(seed)
t.cuda.manual_seed_all(seed)
np.random.seed(seed)

time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S.%f')[:-3]
version_str = time_str[:10]
run_file_name = sys.argv[0].split('/')[-1].split('.')[0]

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='OpenEA')
parser.add_argument('--log', action='store_true')
# ================= Dataset ===============================================
data_group = parser.add_argument_group(title='General Dataset Options')
data_group.add_argument('--result_root', type=str, default='../outputs')
data_group.add_argument('--functionality', action='store_true')
data_group.add_argument('--blocking', action='store_true')
data_group.add_argument('--pretrain_bert_path', type=str)
data_group.add_argument('--basic_bert_path', type=str)
data_group.add_argument('--datasets_root', type=str)
data_group.add_argument('--relation', action='store_true', default=True)
# =========================================================================
# ================= OpenEA ================================================
openea_group = parser.add_argument_group(title='OpenEA Dataset Options')
openea_group.add_argument('--dataset', type=str, metavar='dataset path')
openea_group.add_argument('--fold', type=int, default=0)
# =========================================================================
train_group = parser.add_argument_group(title='Train Options')
train_group.add_argument('--gpus', type=str)
train_group.add_argument('--version', type=str)
# ================= TEA ================================================
tea_group = parser.add_argument_group(title='TEA Options')
tea_group.add_argument('--neighbor', action='store_true')
tea_group.add_argument('--lambda_prompt_ce', type=float, default=1)
tea_group.add_argument('--lambda_prompt_margin', type=float, default=1)
tea_group.add_argument('--template_id', type=int, default=4)
tea_group.add_argument('--nsp', action='store_true', default=False)
tea_group.add_argument('--reranking_cands', type=int, default=256)
tea_group.add_argument('--reranking_thres', type=float, default=0.8)
# =====================================================================
# ================ ME3A ==============================================
train_group.add_argument('--visual', action='store_true')
train_group.add_argument('--prefix', type=int, default=2)
train_group.add_argument('--woattr', action='store_true')
# =====================================================================


args = parser.parse_args()
seq_max_len = 200
bert_output_dim = 300
PARALLEL = True
DEBUG = False
SCORE_DISTANCE_LEVEL, MARGIN = 2, 1
if args.version is not None:
    version_str = args.version
if args.relation:
    version_str += '-relation'

# ================= OpenEA ================================================
dataset_name = args.dataset
dataset_home = os.path.join(args.datasets_root, dataset_name)

ds = dataset_name.split('_')
# result_name = '-'.join((time_str, dataset_name, str(args.fold), run_file_name, str(seq_max_len)))
log_name = '-'.join((time_str, dataset_name, str(args.fold), run_file_name, str(seq_max_len)))
result_home = '/'.join((args.result_root, version_str, dataset_name))
if not os.path.exists(result_home):
    os.makedirs(result_home)
need_log = args.log
log_path = os.path.join(result_home, 'logs')
if need_log:
    Logger.make_print_to_file(name=log_name + '-add.txt', path=log_path)
print("log_path", log_path)
print("log_name", log_name)


class Dataset:
    def __init__(self, no):
        self.name = ds[no - 1]
        self.attr = self.triples('attr', no)
        self.rel = self.triples('rel', no)
        self.entities_out = self.outputs_tab('entities', no)
        self.literals_out = self.outputs_tab('literals', no)
        self.properties_out = self.outputs_tab('properties', no)
        self.relations_out = self.outputs_python('relations', no)
        self.table_out = self.outputs_csv('properties', no)
        # ----------------------- TEA --------------------------------------
        self.attr_seq_out = self.outputs_tab('attr_sequence_form', no)
        self.attr_tokens_out = self.outputs_python('attr_tokens', no)
        self.attr_tids_out = self.outputs_python('attr_tids', no)
        self.attr_token_freqs_out = self.outputs_python('attr_token_freqs', no)

        self.neighboronly_seq_out = self.outputs_tab('neighboronly_sequence_form', no)
        self.neighboronly_tokens_out = self.outputs_python('neighboronly_tokens', no)
        self.neighboronly_tids_out = self.outputs_python('neighboronly_tids', no)
        self.neighboronly_token_freqs_out = self.outputs_python('neighboronly_token_freqs', no)
        # -------------------------------------------------------------
        self.img_feat_out = self.outputs_pickle('img_feat', no)
        # ----------------------------------------------------------------
        self.facts_out = self.outputs_python('facts', no)
        self.case_study_out = self.outputs_python('case_study', no)

    def __str__(self):
        return 'Dataset{name: %s, rel: %s, attr: %s}' % (self.name, self.rel, self.attr)

    @staticmethod
    def triples(name, no):
        file_name = '_'.join((name, 'triples', str(no)))
        return os.path.join(dataset_home, file_name)

    @staticmethod
    def outputs_tab(name, no):
        file_name = '_'.join((name, 'tab', str(no))) + '.txt'
        return os.path.join(result_home, file_name)

    @staticmethod
    def outputs_csv(name, no):
        file_name = '_'.join((name, 'csv', str(no))) + '.csv'
        return os.path.join(result_home, file_name)

    @staticmethod
    def outputs_python(name, no):
        file_name = '_'.join((name, 'python', str(no))) + '.txt'
        return os.path.join(result_home, file_name)

    @staticmethod
    def outputs_pickle(name, no):
        file_name = '_'.join((name, 'pickle', str(no))) + '.pickle'
        return os.path.join(result_home, file_name)



dataset1 = Dataset(1)
dataset2 = Dataset(2)


class OEAlinks:
    def __init__(self, fold):
        self.block = self.result_path('block', 0)
        # if fold > 0:
        self.train = self.links_path('train', fold)
        self.valid = self.links_path('valid', fold)
        self.test = self.links_path('test', fold)
        self.truth = '/'.join((dataset_home, 'ent_links'))
        self.model_save = '/'.join((result_home, log_name, 'basic_bert_model.pkl'))
        self.rel_model_save = '/'.join((result_home, log_name, 'rel_model.pkl'))
        self.case_study_out_1 = '/'.join((result_home, log_name, 'case_study_1.txt'))
        self.case_study_out_2 = '/'.join((result_home, log_name, 'case_study_2.txt'))
        if args.basic_bert_path is None:
            self.kb_prop_emb_1 = '/'.join((result_home, log_name, '_'.join((str(fold), 'kb_prop_emb_1.pt'))))
            self.kb_prop_emb_2 = '/'.join((result_home, log_name, '_'.join((str(fold), 'kb_prop_emb_2.pt'))))
        else:
            self.kb_prop_emb_1 = '/'.join(
                (os.path.dirname(args.basic_bert_path), '_'.join((str(fold), 'kb_prop_emb_1.pt'))))
            self.kb_prop_emb_2 = '/'.join(
                (os.path.dirname(args.basic_bert_path), '_'.join((str(fold), 'kb_prop_emb_2.pt'))))
            self.rel_model_load = '/'.join((os.path.dirname(args.basic_bert_path), 'rel_model.pkl'))
        pass

    @staticmethod
    def links_path(train_type, fold_num):
        return os.path.join(dataset_home, '/'.join(('721_5fold', str(fold_num), '_'.join((train_type, 'links')))))

    @staticmethod
    def result_path(name, fold_num):
        return os.path.join(result_home, '_'.join((name, str(fold_num), '.txt')))


links = OEAlinks(args.fold)

if args.mode == 'KB':
    functionality_control = True
else:
    functionality_control = args.functionality
functionality_threshold = 0.9

print('time str:', time_str)
print('run file:', run_file_name)
print('args:')
print(args)
print('log path:', os.path.abspath(log_path))
print('log:', need_log)

print('dataset1:', dataset1)
print('dataset2:', dataset2)
# print('result_name:', result_name)
print('result_path:', result_home)

# ========================================================

# ===================== TEA Config =======================

PROMPT_MARGIN = 1

templates_list = [
    "?[SEP] [MASK].",
    "?[SEP] [MASK]. I know that ",
    "?[SEP] [MASK]. I think that ",
    "[SEP][MASK].[P0][P1]",
    "[SEP][MASK].[P0][P1][P2]",
    "[SEP][MASK].[P0][P1][P2][P3]",
    "[SEP][MASK].[P0][P1][P2][P3][P4]"
]

prompt_labels = ['No', 'Yes']
template = templates_list[args.template_id]

num_prompt_labels = len(prompt_labels)

# special_tokens_dict = {'additional_special_tokens': ['[P0]', '[P1]', '[P2]', '[P3]', '[P4]', '[P5]']}
special_tokens_dict = {'additional_special_tokens': ['[VP]', '[P0]', '[P1]', '[P2]', '[P3]',
                                                     '[P4]', '[P5]', '[PS]', '[PE]']}

tokenizer = BertTokenizer.from_pretrained(args.pretrain_bert_path)
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
len_tokenizer = len(tokenizer)
print("special_tokens_dict", special_tokens_dict)
print("num_added_toks", num_added_toks)
print("len_tokenizer", len_tokenizer)

if not args.visual:
    args.prefix = 0
visual_hidden_size = 4096 if args.dataset.startswith('fb') else 2048

# =================================================
