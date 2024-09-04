import torch as t
import pickle
from config.KBConfig import *
from preprocess.BertDataLoader import BertDataLoader
from preprocess.KBStore import KBStore
from tools.Announce import Announce
from train.PairwiseTrainer import PairwiseTrainer
from train.PairwisePromptTrainer import PairwisePromptTrainer
from train.MMPairwisePromptTrainer import MMPairwisePromptTrainer

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # device = t.device("cuda" if t.cuda.is_available() else "cpu")
    device = t.device('cuda')
    print(device)

    attr_tids1 = BertDataLoader.load_saved_type_seq(dataset1, 'attr')
    attr_tids2 = BertDataLoader.load_saved_type_seq(dataset2, 'attr')
    max_len1 = max([len(tokens) for eid, tokens in attr_tids1])
    max_len2 = max([len(tokens) for eid, tokens in attr_tids2])
    print(Announce.printMessage(), 'Max len 1:', max_len1)
    print(Announce.printMessage(), 'Max len 2:', max_len2)
    eid2attr_tids1 = {eid: tids for eid, tids in attr_tids1}
    eid2attr_tids2 = {eid: tids for eid, tids in attr_tids2}
    fs1 = KBStore(dataset1)
    fs2 = KBStore(dataset2)
    fs1.load_kb_from_saved()
    fs2.load_kb_from_saved()

    nei_tids1 = BertDataLoader.load_saved_type_seq(dataset1, 'neighboronly')
    nei_tids2 = BertDataLoader.load_saved_type_seq(dataset2, 'neighboronly')
    eid2img_feat1 = pickle.load(open(dataset1.img_feat_out, 'rb')) if args.visual else None
    eid2img_feat2 = pickle.load(open(dataset2.img_feat_out, 'rb')) if args.visual else None

    eid2nei_tids1 = {eid: tids for eid, tids in nei_tids1}
    eid2nei_tids2 = {eid: tids for eid, tids in nei_tids2}

    trainer = MMPairwisePromptTrainer()
    if args.woattr:
        args.neighbor = False
        trainer.data_prepare(eid2nei_tids1, eid2nei_tids2, fs1, fs2)
    else:
        trainer.data_prepare(eid2attr_tids1, eid2attr_tids2, fs1, fs2,
                             eid2nei_tids1, eid2nei_tids2,
                             eid2img_feat1, eid2img_feat2)

    trainer.train(device=device)
