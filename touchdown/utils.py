import json, re, string
import numpy as np
import os, sys, torch
import warnings
from tensorboardX import SummaryWriter
import networkx as nx
import random
import shutil


base_vocab = ['<PAD>', '<START>', '<EOS>', '<UNK>']
padding_idx = 0


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def read_vocab(path):
    with open(path, encoding="utf-8") as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


class Tokenizer(object):
    """ Class to tokenize and encode a sentence. """
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character

    def __init__(self, remove_punctuation=False, reversed=True, vocab=None, encoding_length=20):
        self.remove_punctuation = remove_punctuation
        self.reversed = reversed
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.table = str.maketrans({key: None for key in string.punctuation})
        self.word_to_index = {}
        if vocab:
            for i, word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        """ Break sentence into a list of words and punctuation """
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if
                     len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []

        splited = self.split_sentence(sentence)
        if self.reversed:
            splited = splited[::-1]

        if self.remove_punctuation:
            splited = [word for word in splited if word not in string.punctuation]

        for word in splited:  # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])

        encoding.append(self.word_to_index['<EOS>'])
        encoding.insert(0, self.word_to_index['<START>'])

        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length - len(encoding))
        return np.array(encoding[:self.encoding_length])

    def encode_instructions(self, instructions):
        rst = []
        for sent in instructions.strip().split('. '):
            rst.append(self.encode_sentence(sent))
        return rst


def load_datasets(splits, opts=None):
    data = []
    for split in splits:
        assert split in ['train', 'test', 'dev']
        with open('%s/data/%s.json' % (opts.dataset, split)) as f:
            for line in f:
                data.append(json.loads(line))
    return data


def shortest_path(pano, Graph):
    dis = {}
    queue = []
    queue.append([Graph.graph.nodes[pano], 0])
    while queue:
        cur = queue.pop(0)
        cur_node = cur[0]
        cur_dis = cur[1]
        if cur_node.panoid not in dis.keys():
            dis[cur_node.panoid] = cur_dis
            cur_dis += 1
            for neighbors in cur_node.neighbors.values():
                queue.append([neighbors, cur_dis])
                
    with open("path/"+pano+".json", "a") as f:
        json.dump(dis, f)

        
def resume_training(opts, model, instr_encoder, optimizer, text_linear=None, pano_encoder=None, bert_optimizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if opts.resume == 'latest':
        file_extention = '.pth.tar'
    elif opts.resume == 'SPD_best':
        file_extention = '_model_SPD_best.pth.tar'
    elif opts.resume == 'TC_best':
        file_extention = '_model_TC_best.pth.tar'
    else:
        raise ValueError('Unknown resume option: {}'.format(opts.resume))
    exp_name = opts.resume_from if opts.resume_from is not None else opts.exp_name
    opts.resume = ('{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, exp_name, file_extention))
    if os.path.isfile(opts.resume):
        checkpoint = torch.load(opts.resume, map_location=device)
        opts.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        instr_encoder.load_state_dict(checkpoint['instr_encoder_state_dict'])
        if opts.resume_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Optimizer resumed, lr = %f' % optimizer.param_groups[0]['lr'])
        if opts.model == 'vlntrans':
            text_linear.load_state_dict(checkpoint['text_linear_state_dict'])
            pano_encoder.load_state_dict(checkpoint['pano_encoder_state_dict'])
            if opts.resume_optimizer:
                bert_optimizer.load_state_dict(checkpoint['bert_optimizer'])
                print('BERT Optimizer resumed, bert_lr = %f' % bert_optimizer.param_groups[0]['lr'])
        try:
            best_SPD = checkpoint['best_SPD']
        except KeyError:
            print('best_SPD not provided in ckpt, set to inf.')
            best_SPD = float('inf')
        try:
            best_TC = checkpoint['best_TC']
        except KeyError:
            print('best_TC not provided in ckpt, set to 0.0.')
            best_TC = 0.0
        print("=> loaded checkpoint '{}' (epoch {})".format(opts.resume, checkpoint['epoch']-1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(opts.resume))
    if opts.model == 'vlntrans':
        return model, instr_encoder, text_linear, pano_encoder, optimizer, bert_optimizer, best_SPD, best_TC
    else:
        return model, instr_encoder, optimizer, best_SPD, best_TC


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
    

def set_tb_logger(log_dir, exp_name, resume):
    """ Set up tensorboard logger"""
    log_dir = log_dir + '/' + exp_name
    # remove previous log with the same name, if not resume
    if not resume and os.path.exists(log_dir):
        import shutil
        try:
            shutil.rmtree(log_dir)
        except:
            warnings.warn('Experiment existed in TensorBoard, but failed to remove')
    return SummaryWriter(log_dir=log_dir)


def load_nav_graph(opts):
    with open("%s/graph/links.txt" % opts.dataset) as f:
        G = nx.Graph()
        for line in f:
            pano_1, _, pano_2 = line.strip().split(",")
            G.add_edge(pano_1, pano_2)        
    return G


def random_list(prob_torch, lists):
    x = random.uniform(0, 1)
    cum_prob = 0
    for i in range(len(lists) - 1):
        cum_prob += prob_torch[i]
        if x < cum_prob:
            return lists[i]
    return lists[len(lists) - 1]
    
    
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best_SPD, is_best_TC, epoch=-1):
    opts = state['opts']
    os.makedirs('{}/{}/{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name), exist_ok=True)
    filename = ('{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '.pth.tar'))
    if opts.store_ckpt_every_epoch:
        filename = ('{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '.%d.pth.tar' % epoch))
    torch.save(state, filename)
    if is_best_SPD:
        best_filename = (
            '{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '_model_SPD_best.pth.tar'))
        shutil.copyfile(filename, best_filename)
    if is_best_TC:
        best_filename = (
            '{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '_model_TC_best.pth.tar'))
        shutil.copyfile(filename, best_filename)


def input_img(pano, path):
    return np.load(path+"/"+pano+".npy")
