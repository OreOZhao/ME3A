# from .PairwisePromptTrainer import PairwisePromptTrainer
from config.KBConfig import *


class MMPairwisePromptDataset(Dataset):
    def __init__(self, train_tups, ent2attr1, ent2attr2,
                 temp_tids, cls_token, sep_token, vis_token,
                 ent2nei1=None, ent2nei2=None,
                 ent2vis1=None, ent2vis2=None,
                 ent2name1=None, ent2name2=None):
        self.train_tups = train_tups
        self.ent2data1 = ent2attr1  # eid2tids1
        self.ent2data2 = ent2attr2  # eid2tids2
        self.ent2nei1 = ent2nei1
        self.ent2nei2 = ent2nei2
        self.ent2vis1 = ent2vis1
        self.ent2vis2 = ent2vis2
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.vis_token = vis_token
        self.pad_token = 0
        self.temp_tids = temp_tids  # if name is not None, [tids list of prompt_template_with_name], else tids
        self.ent2name1 = ent2name1
        self.ent2name2 = ent2name2
        if ent2vis1 == None and ent2vis2 == None:
            self.prefix = 0
        else:
            self.prefix = args.prefix

    def __len__(self):
        return len(self.train_tups)

    def __getitem__(self, item):
        pe1, pe2, ne1, ne2 = self.train_tups[item]
        return self.seq_generator(self.ent2data1[pe1], self.ent2data2[pe2],
                                  self.ent2nei1[pe1] if self.ent2nei1 is not None else None,
                                  self.ent2nei2[pe2] if self.ent2nei2 is not None else None,
                                  self.ent2vis1[pe1] if self.ent2vis1 is not None else None,
                                  self.ent2vis2[pe2] if self.ent2vis1 is not None else None), \
            self.seq_generator(self.ent2data2[pe2], self.ent2data1[pe1],
                               self.ent2nei2[pe2] if self.ent2nei1 is not None else None,
                               self.ent2nei1[pe1] if self.ent2nei1 is not None else None,
                               self.ent2vis2[pe2] if self.ent2vis1 is not None else None,
                               self.ent2vis1[pe1] if self.ent2vis1 is not None else None), \
            self.seq_generator(self.ent2data1[ne1], self.ent2data2[ne2],
                               self.ent2nei1[ne1] if self.ent2nei1 is not None else None,
                               self.ent2nei2[ne2] if self.ent2nei1 is not None else None,
                               self.ent2vis1[ne1] if self.ent2vis1 is not None else None,
                               self.ent2vis2[ne2] if self.ent2vis1 is not None else None), \
            self.seq_generator(self.ent2data2[ne2], self.ent2data1[ne1],
                               self.ent2nei2[ne2] if self.ent2nei1 is not None else None,
                               self.ent2nei1[ne1] if self.ent2nei1 is not None else None,
                               self.ent2vis2[ne2] if self.ent2vis1 is not None else None,
                               self.ent2vis1[ne1] if self.ent2vis1 is not None else None)
        # return self.ent2data1[pe1], self.ent2data2[pe2], self.ent2data1[ne1], self.ent2data2[ne2]

    def seq_generator(self, tids1, tids2, ntids1=None, ntids2=None, vis1=None, vis2=None):
        assert len(tids1) > 0  # input tids have no CLS or SEP
        assert len(tids2) > 0
        my_max_len = (seq_max_len - len(self.temp_tids) - 2 * self.prefix) / 2
        if ntids1 is not None and ntids2 is not None:
            tids1 = self.reduce_tokens(tids1, max_len=my_max_len // 2)  # 减少token
            tids2 = self.reduce_tokens(tids2, max_len=my_max_len // 2)  # 减少token
            ntids1 = self.reduce_tokens(ntids1, max_len=my_max_len // 2)  # 减少token
            ntids2 = self.reduce_tokens(ntids2, max_len=my_max_len // 2)  # 减少token
            tids1.extend(ntids1)
            tids2.extend(ntids2)
        tids1 = self.reduce_tokens(tids1, max_len=my_max_len)  # 减少token
        tids2 = self.reduce_tokens(tids2, max_len=my_max_len)  # 减少token
        tids1 = [self.vis_token] * args.prefix + tids1
        tids2 = [self.vis_token] * args.prefix + tids2
        prompt_tids = tids1 + self.temp_tids + tids2
        pad_length = seq_max_len - len(prompt_tids)
        input_ids = [self.cls_token] + [self.cls_token] + prompt_tids + [self.pad_token] * pad_length  # two cls
        masks1 = [1, 1] + [1] * len(prompt_tids) + [self.pad_token] * pad_length
        masks2 = [1, 0] + [1] * len(tids1) + [self.pad_token] * (len(self.temp_tids + tids2) + pad_length)
        masks3 = [0, 1] + [0] * len(tids1 + self.temp_tids) + [1] * len(tids2) + [self.pad_token] * pad_length
        assert len(input_ids) == seq_max_len + 2, len(input_ids)
        assert len(masks1) == seq_max_len + 2, len(input_ids)
        assert len(masks2) == seq_max_len + 2, len(input_ids)
        assert len(masks3) == seq_max_len + 2, len(input_ids)

        # input_ids = np.array(input_ids, dtype=np.long)
        # masks = np.array(masks, dtype=np.long)
        input_ids = t.tensor(input_ids, dtype=t.long)
        masks = t.tensor([masks1, masks2, masks3], dtype=t.long)
        if self.prefix > 0:
            return input_ids, masks, t.tensor(vis1).float(), t.tensor(vis2).float()
        else:
            return input_ids, masks

    def reduce_tokens(self, tids, max_len=200):
        while True:
            total_length = len(tids)
            if total_length <= max_len:
                break
            tids.pop()
        return tids


class MMPairwisePromptTestDataset(Dataset):
    def __init__(self, test_tups, candidates, ent2data1, ent2data2,
                 temp_tids, cls_token, sep_token, vis_token,
                 ent2nei1=None, ent2nei2=None,
                 ent2vis1=None, ent2vis2=None,
                 train=False, ent2name1=None, ent2name2=None):
        self.test_tups = test_tups  # pe1, pe2
        self.candidates = candidates
        self.ent2data1 = ent2data1  # eid2tids1
        self.ent2data2 = ent2data2  # eid2tids2
        self.ent2nei1 = ent2nei1
        self.ent2nei2 = ent2nei2
        self.ent2vis1 = ent2vis1
        self.ent2vis2 = ent2vis2
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = 0
        self.temp_tids = temp_tids
        self.train = train
        self.ent2name1 = ent2name1
        self.ent2name2 = ent2name2
        self.vis_token = vis_token
        if ent2vis1 == None and ent2vis2 == None:
            self.prefix = 0
        else:
            self.prefix = args.prefix

    def __len__(self):
        return len(self.test_tups)

    def __getitem__(self, item):
        if self.train:
            pe1, pe2 = self.test_tups[item]
            candidates = self.candidates[item]  # 6443

            tids, masks = self.seq_generator(self.ent2data1[pe1], self.ent2data2[int(pe2)],
                                             self.ent2nei1[pe1] if self.ent2nei1 is not None else None,
                                             self.ent2nei2[pe2] if self.ent2nei2 is not None else None,
                                             self.ent2vis1[pe1] if self.ent2vis1 is not None else None,
                                             self.ent2vis2[pe2] if self.ent2vis1 is not None else None)
            seq_tids = [tids.unsqueeze(0)]
            seq_masks = [masks.unsqueeze(0)]
            tids_r, masks_r = self.seq_generator(self.ent2data2[int(pe2)], self.ent2data1[pe1],
                                                 self.ent2nei2[pe2] if self.ent2nei1 is not None else None,
                                                 self.ent2nei1[pe1] if self.ent2nei2 is not None else None,
                                                 self.ent2vis2[pe2] if self.ent2vis1 is not None else None,
                                                 self.ent2vis1[pe1] if self.ent2vis1 is not None else None)
            seq_tids_r = [tids_r.unsqueeze(0)]
            seq_masks_r = [masks_r.unsqueeze(0)]
            for ce2 in candidates:
                ce2 = int(ce2)
                tids, masks = self.seq_generator(self.ent2data1[pe1], self.ent2data2[int(ce2)],
                                                 self.ent2nei1[pe1] if self.ent2nei1 is not None else None,
                                                 self.ent2nei2[ce2] if self.ent2nei2 is not None else None,
                                                 self.ent2vis1[pe1] if self.ent2vis1 is not None else None,
                                                 self.ent2vis2[ce2] if self.ent2vis1 is not None else None)
                seq_tids.append(tids.unsqueeze(0))
                seq_masks.append(masks.unsqueeze(0))
                tids_r, masks_r = self.seq_generator(self.ent2data2[int(ce2)], self.ent2data1[pe1],
                                                     self.ent2nei2[ce2] if self.ent2nei1 is not None else None,
                                                     self.ent2nei1[pe1] if self.ent2nei2 is not None else None,
                                                     self.ent2vis2[ce2] if self.ent2vis1 is not None else None,
                                                     self.ent2vis1[pe1] if self.ent2vis1 is not None else None)
                seq_tids_r.append(tids_r.unsqueeze(0))
                seq_masks_r.append(masks_r.unsqueeze(0))
            seq_tids = t.cat(seq_tids, 0)
            seq_masks = t.cat(seq_masks, 0)
            seq_tids_r = t.cat(seq_tids_r, 0)
            seq_masks_r = t.cat(seq_masks_r, 0)
            return seq_tids, seq_masks, seq_tids_r, seq_masks_r
        else:
            pe1, pe2 = self.test_tups[item]
            candidates = self.candidates[item]
            # else:
            seq_tids, seq_masks, seq_vis1s, seq_vis2s = [], [], [], []
            seq_tids_r, seq_masks_r, seq_vis1s_r, seq_vis2s_r = [], [], [], []

            for ce2 in candidates:
                seq_tid, seq_mask, seq_vis1, seq_vis2 = self.seq_generator(
                    self.ent2data1[pe1], self.ent2data2[int(ce2)],
                    self.ent2nei1[pe1] if self.ent2nei1 is not None else None,
                    self.ent2nei2[int(ce2)] if self.ent2nei2 is not None else None,
                    self.ent2vis1[pe1] if self.ent2vis1 is not None else None,
                    self.ent2vis2[int(ce2)] if self.ent2vis1 is not None else None
                )
                seq_tid_r, seq_mask_r, seq_vis1_r, seq_vis2_r = self.seq_generator(
                    self.ent2data2[int(ce2)], self.ent2data1[pe1],
                    self.ent2nei2[int(ce2)] if self.ent2nei1 is not None else None,
                    self.ent2nei1[pe1] if self.ent2nei2 is not None else None,
                    self.ent2vis2[int(ce2)] if self.ent2vis1 is not None else None,
                    self.ent2vis1[pe1] if self.ent2vis1 is not None else None
                )
                seq_tids.append(seq_tid.unsqueeze(0))
                seq_masks.append(seq_mask.unsqueeze(0))
                seq_vis1s.append(seq_vis1.unsqueeze(0))
                seq_vis2s.append(seq_vis2.unsqueeze(0))
                seq_tids_r.append(seq_tid_r.unsqueeze(0))
                seq_masks_r.append(seq_mask_r.unsqueeze(0))
                seq_vis1s_r.append(seq_vis1_r.unsqueeze(0))
                seq_vis2s_r.append(seq_vis2_r.unsqueeze(0))
            seq_tids = t.cat(seq_tids, 0)
            seq_masks = t.cat(seq_masks, 0)
            seq_vis1s = t.cat(seq_vis1s, 0)
            seq_vis2s = t.cat(seq_vis2s, 0)
            seq_tids_r = t.cat(seq_tids_r, 0)
            seq_masks_r = t.cat(seq_masks_r, 0)
            seq_vis1s_r = t.cat(seq_vis1s_r, 0)
            seq_vis2s_r = t.cat(seq_vis2s_r, 0)
            return seq_tids, seq_masks, seq_vis1s, seq_vis2s, seq_tids_r, seq_masks_r, seq_vis1s_r, seq_vis2s_r


    def seq_generator(self, tids1, tids2, ntids1=None, ntids2=None, vis1=None, vis2=None):
        assert len(tids1) > 0  # input tids have no CLS or SEP
        assert len(tids2) > 0
        my_max_len = (seq_max_len - len(self.temp_tids) - 2 * self.prefix) / 2
        if ntids1 is not None and ntids2 is not None:
            tids1 = self.reduce_tokens(tids1, max_len=my_max_len // 2)  # 减少token
            tids2 = self.reduce_tokens(tids2, max_len=my_max_len // 2)  # 减少token
            ntids1 = self.reduce_tokens(ntids1, max_len=my_max_len // 2)  # 减少token
            ntids2 = self.reduce_tokens(ntids2, max_len=my_max_len // 2)  # 减少token
            tids1.extend(ntids1)
            tids2.extend(ntids2)
        tids1 = self.reduce_tokens(tids1, max_len=my_max_len)  # 减少token
        tids2 = self.reduce_tokens(tids2, max_len=my_max_len)  # 减少token
        tids1 = [self.vis_token] * args.prefix + tids1
        tids2 = [self.vis_token] * args.prefix + tids2
        prompt_tids = tids1 + self.temp_tids + tids2
        pad_length = seq_max_len - len(prompt_tids)
        input_ids = [self.cls_token] + [self.cls_token] + prompt_tids + [self.pad_token] * pad_length  # two cls
        masks1 = [1, 1] + [1] * len(prompt_tids) + [self.pad_token] * pad_length
        masks2 = [1, 0] + [1] * len(tids1) + [self.pad_token] * (len(self.temp_tids + tids2) + pad_length)
        masks3 = [0, 1] + [0] * len(tids1 + self.temp_tids) + [1] * len(tids2) + [self.pad_token] * pad_length
        assert len(input_ids) == seq_max_len + 2, len(input_ids)
        assert len(masks1) == seq_max_len + 2, len(input_ids)
        assert len(masks2) == seq_max_len + 2, len(input_ids)
        assert len(masks3) == seq_max_len + 2, len(input_ids)

        # input_ids = np.array(input_ids, dtype=np.long)
        # masks = np.array(masks, dtype=np.long)
        input_ids = t.tensor(input_ids, dtype=t.long)
        masks = t.tensor([masks1, masks2, masks3], dtype=t.long)
        if self.prefix > 0:
            return input_ids, masks, t.tensor(vis1).float(), t.tensor(vis2).float()
        else:
            return input_ids, masks


    def reduce_tokens(self, tids, max_len=200):
        while True:
            total_length = len(tids)
            if total_length <= max_len:
                break
            tids.pop()
        return tids
