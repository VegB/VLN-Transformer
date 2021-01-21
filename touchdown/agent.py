import random, torch
from torch import nn
import numpy as np
from utils import padding_idx


class BaseAgent:
    def __init__(self, env):
        self.env = env
        random.seed(1)

    def _get_tensor_and_length(self, encodings):
        """
        :param encodings: a list of 1-d numpy arrays, K elements
        :return: 2-d tensor of size (K, max_seq_len), and a 1-d list
        """
        seq_tensor = np.array(encodings)
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length
        seq_tensor = torch.from_numpy(seq_tensor)
        return seq_tensor.long().to(self.device), list(seq_lengths)

    def _sort_batch(self):
        """ Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). """
        seq_tensor = np.array([item['instr_encoding'] for item in self.env.batch])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length
        seq_tensor = torch.from_numpy(seq_tensor)
        return seq_tensor.long().to(self.device), list(seq_lengths)

    def _vlntrans_sort_batch(self):
        """ getting sequence and corresponding lengths """
        # return self._get_tensor_and_length([item['instr_encoding'] for item in self.env.batch])
        input_ids, segment_ids, input_mask, sentence_ids, sentence_num = [], [], [], [], []
        for item in self.env.batch:
            input_ids.append(item['encoder_input'][0])
            segment_ids.append(item['encoder_input'][1])
            input_mask.append(item['encoder_input'][2])
            sentence_ids.append(item['encoder_input'][3])
            sentence_num.append(item['encoder_input'][4])
        return torch.LongTensor(input_ids).to(self.device), \
               torch.LongTensor(segment_ids).to(self.device), \
               torch.LongTensor(input_mask).to(self.device), \
               torch.LongTensor(sentence_ids).to(self.device), sentence_num

    def _concat_textual_visual(self, textual_vectors, visual_vectors, sentence_nums, max_length):
        """
        :param textual_vectors: [batch_size, current_max_sent_num, hidden_dim]
        :param visual_vectors: [batch_size, n, hidden_dim], in which n is increasing
        :param sentence_nums: a list with batch_size elements
        :param max_length: an int
        :return: t_v_embeds, lengths, segment_ids
        """
        batch_size, pano_num, hidden_dim = visual_vectors.shape
        max_sent_num = textual_vectors.shape[1]
        t_v_embs, lengths, segment_ids = [], [], []
        for idx, sent_num in enumerate(sentence_nums):
            sent_num = sent_num if sent_num <= max_sent_num else max_sent_num
            pad_len = max_length - sent_num - pano_num
            cur_pano_num = pano_num
            if pad_len < 0:
                pad_len = 0
                cur_pano_num = max_length - sent_num
            pad_vec = torch.zeros(1, pad_len, hidden_dim).to(self.device)
            t_v_emb = torch.cat((textual_vectors[idx][:sent_num].unsqueeze(dim=0),
                                 visual_vectors[idx][-cur_pano_num:].unsqueeze(dim=0), pad_vec), dim=1)  # [1, max_length, hidden_dim]
            assert t_v_emb.shape[1] == max_length
            t_v_embs.append(t_v_emb)
            lengths.append(sent_num+cur_pano_num)
            seg_ids = torch.cat((torch.zeros(1, sent_num),
                                 torch.ones(1, cur_pano_num),
                                 torch.zeros(1, pad_len)), dim=1).long().to(self.device)
            segment_ids.append(seg_ids)
        return torch.cat(tuple(t_v_embs), dim=0), \
               torch.LongTensor(lengths).to(self.device), \
               torch.cat(tuple(segment_ids), dim=0).to(self.device)


class TouchdownVLNTransformer(BaseAgent):
    def __init__(self, opts, env, instr_encoder, pano_encoder, text_linear, model):
        super(TouchdownVLNTransformer, self).__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts
        self.instr_encoder = instr_encoder
        self.text_linear = text_linear
        self.pano_encoder = pano_encoder
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=4)

    def _encode_instruction(self, input_ids, segment_ids, input_mask):
        """input_ids, segment_ids, input_mask: [batch_size, 512]"""
        input_length = (1 - (input_ids == 0).int()).sum(dim=1)
        encoded_word_embs, _ = self.instr_encoder(input_ids, input_length, segment_ids)  # [batch_size, max_sent_len, 768]
        mask = input_mask.unsqueeze(-1).expand_as(encoded_word_embs).float()
        encoded_word_embs = torch.mean(encoded_word_embs.mul(mask), dim=1).unsqueeze(dim=1)  # [batch_size, 1, 768]
        return self.text_linear(encoded_word_embs)  # [batch_size, 1, hidden_dim]

    def _encode_sentences(self, input_ids, segment_ids, sentence_ids):
        """input_ids, segment_ids, sentence_lengths: [batch_size, 512]"""
        input_length = (1 - (input_ids == 0).int()).sum(dim=1)
        encoded_word_embs, _ = self.instr_encoder(input_ids, input_length, segment_ids)  # [batch_size, max_sent_len, 768]
        max_sent_len = sentence_ids.max().item()
        encoded_sent_embs = []
        for i in range(max_sent_len):
            mask = (sentence_ids == i)  # [batch_size, 512]
            sent_len = mask.sum(dim=1).unsqueeze(dim=1)  # [batch_size]
            mask = mask.unsqueeze(-1).expand_as(encoded_word_embs).float()
            embs = encoded_word_embs.mul(mask).sum(dim=1)  # [batch_size, 768]
            sent_len = sent_len.expand_as(embs).float() + 1e-13  # [batch_size, 768]
            embs = embs.div(sent_len).unsqueeze(dim=1)
            encoded_sent_embs.append(embs)
        encoded_sent_embs = torch.cat(tuple(encoded_sent_embs), dim=1)
        return self.text_linear(encoded_sent_embs)  # [batch_size, current_max_sent_len, hidden_dim]

    def rollout(self, is_test):
        trajs = self.env.reset()  # a batch of the first panoid for each route_panoids
        batch_size = len(self.env.batch)
        input_ids, text_segment_ids, input_mask, sentence_ids, sentence_num = self._vlntrans_sort_batch()
        encoded_texts = self._encode_sentences(input_ids, text_segment_ids, sentence_ids)  # [batch_size, cur_max_sent_len, hidden_dim]
        ended = torch.BoolTensor([0] * batch_size).to(self.device)
        num_act_nav = [batch_size]
        loss = 0
        total_steps = [0]
        encoded_panos = None
        for step in range(self.opts.max_route_len):
            I = self.env.get_imgs()
            I = self.pano_encoder(I).unsqueeze(1)  # [batch_size, 1, 256]
            if encoded_panos is None:
                encoded_panos = I
            else:
                encoded_panos = torch.cat((encoded_panos, I), dim=1)
            t_v_embeds, lengths, segment_ids = \
                self._concat_textual_visual(encoded_texts, encoded_panos, sentence_num, self.opts.max_t_v_len)
            logits, preds, _ = self.model(t_v_embeds, lengths, segment_ids)
            if is_test:
                self.env.action_select(logits, ended, num_act_nav, trajs, total_steps)
            else:
                target = self.env.get_gt_action(is_test)
                target_ = target.masked_fill(ended, value=torch.tensor(4))
                loss += self.criterion(logits, target_) * num_act_nav[0]
                self.env.env.action_step(target, ended, num_act_nav, trajs, total_steps)
                target.unsqueeze(1)
            if not num_act_nav[0]:
                break
        loss /= total_steps[0]
        return loss, trajs


class TouchdownRConcat(BaseAgent):
    def __init__(self, opts, env, encoder, model):
        super(TouchdownRConcat, self).__init__(env)
        self.opts = opts
        self.instr_encoder = encoder
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(ignore_index=4)

    def rollout(self, is_test):
        trajs = self.env.reset()  # a batch of the first panoid for each route_panoids
        batch_size = len(self.env.batch)
        seq, seq_lengths = self._sort_batch()
        x = self.instr_encoder(seq, seq_lengths)  # LSTM encoded hidden states for instructions, [batch_size, 1, 256]
        ended = [0] * batch_size
        ended = torch.BoolTensor(ended).to(self.device)
        h_t = torch.zeros(1, batch_size, 256).to(self.device)
        c_t = torch.zeros(1, batch_size, 256).to(self.device)
        a = torch.ones(batch_size, 1).long().to(self.device)
        t = torch.LongTensor([-1]).to(self.device)
        num_act_nav = [batch_size]
        loss = 0
        total_steps = [0]
        for step in range(self.opts.max_route_len):
            I = self.env.get_imgs()
            t = t + 1
            a_t, (h_t, c_t) = self.model(x, I, a, h_t, c_t, t)  # a_t is a distribution over all actions, [batchsize, 4]
            if is_test:
                a = self.env.action_select(a_t, ended, num_act_nav, trajs, total_steps)
            else:
                target = self.env.get_gt_action(is_test)
                target_ = target.masked_fill(ended, value=torch.tensor(4))
                loss += self.criterion(a_t, target_) * num_act_nav[0]
                self.env.env.action_step(target, ended, num_act_nav, trajs, total_steps)
                a = target.unsqueeze(1)
            if not num_act_nav[0]:
                break
        loss /= total_steps[0]
        return loss, trajs


class TouchdownGA(BaseAgent):
    def __init__(self, opts, env, encoder, model):
        super(TouchdownGA, self).__init__(env)
        self.opts = opts
        self.instr_encoder = encoder
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(ignore_index=4)

    def rollout(self, is_test):
        trajs = self.env.reset()
        batch_size = len(self.env.batch)
        seq, seq_lengths = self._sort_batch()
        x = self.instr_encoder(seq, seq_lengths)
        ended = [0] * batch_size
        ended = torch.BoolTensor(ended).to(self.device)
        h_t = torch.randn(1, batch_size, 256).to(self.device)
        c_t = torch.randn(1, batch_size, 256).to(self.device)
        t = torch.LongTensor([-1]).to(self.device)
        num_act_nav = [batch_size]
        loss = 0
        total_steps = [0]
        for step in range(self.opts.max_route_len):
            I = self.env.get_imgs()
            t = t + 1
            a_t, (h_t, c_t) = self.model(x, I, h_t, c_t, t)
            if is_test:
                a = self.env.action_select(a_t, ended, num_act_nav, trajs, total_steps)
            else:
                target = self.env.get_gt_action(is_test)
                target_ = target.masked_fill(ended, value=torch.tensor(4))
                loss += self.criterion(a_t, target_) * num_act_nav[0]
                self.env.env.action_step(target, ended, num_act_nav, trajs, total_steps)
            if not num_act_nav[0]:
                break
        loss /= total_steps[0]
        return loss, trajs
