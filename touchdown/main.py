from utils import setup_seed
import torch
import torch.nn as nn
import texar.torch as tx
from utils import read_vocab, Tokenizer, resume_training, set_tb_logger, save_checkpoint
from env import load_features, TouchdownBatch
from agent import TouchdownRConcat, TouchdownVLNTransformer, TouchdownGA
from trainer import TouchdownTrainer
from models.GA import GA
from models.VLNTransformer import VLNTransformer
from models.RConcat import Embed_RNN, RConcat, Conv_net
import argparse

parser = argparse.ArgumentParser(description='PyTorch for Touchdown baseline')
parser.add_argument('--model', default='vlntrans', type=str, choices=['rconcat', 'vlntrans', 'ga'])
parser.add_argument('--dataset', default='touchdown', type=str)
parser.add_argument('--img_feat_dir', default='', type=str, help='Path to pre-cached image features.')
parser.add_argument('--log_dir', default='tensorboard_logs/touchdown', type=str, help='Path to tensorboard log files.')
parser.add_argument('--checkpoint_dir', default='checkpoints', type=str, help='Path to the checkpoint dir.')
parser.add_argument('--resume', default='', type=str, choices=['latest', 'TC_best', 'SPD_best'])
parser.add_argument('--resume_from', default=None, type=str, help='resume from other experiment')
parser.add_argument('--store_ckpt_every_epoch', default=False, type=bool)
parser.add_argument('--ckpt_epoch', default=-1, type=int)
parser.add_argument('--test', default=False, type=bool, help='No training. Resume from a model and run testing.')
parser.add_argument('--seed', default=10, type=int, help='random seed')
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--max_num_epochs', default=100, type=int, help='Max training epoch.')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--eval_every_epochs', default=1, type=int, help='How often do we eval the trained model.')
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--CLS', default=False, type=bool, help='Calculate CLS when evaluating.')
parser.add_argument('--DTW', default=False, type=bool, help='calculate DTW when evaluating.')
parser.add_argument('--lr', default=0.00025, type=float)
parser.add_argument('--finetune_bert', default=False, type=bool)
parser.add_argument('--bert_lr', default=0.00001, type=float)
parser.add_argument('--resume_optimizer', default=False, type=bool)
parser.add_argument('--max_instr_len', default=540, type=int, help='Max instruction token num.')
parser.add_argument('--max_sentence_num', default=40, type=int, help='Max number of sentences in instruction.')
parser.add_argument('--max_route_len', default=55, type=int, help='Max trajectory length.')
parser.add_argument('--max_t_v_len', default=60, type=int,
                    help='Max length of the concatenation of sentence embeddings and trajectory embeddings.')
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--exp_name', default='experiments', type=str,
                    help='Name of the experiment. It decides where to store samples and models')
parser.add_argument('--exp_number', default=None, type=str)
opts = parser.parse_args()

setup_seed(opts.seed)


def main(opts):
    if opts.exp_number is not None:
        opts.exp_name = opts.exp_name + '_' + opts.exp_number
    opts.dataset = 'datasets/%s' % opts.dataset
    tb_logger = set_tb_logger('{}/{}'.format(opts.log_dir, opts.model), opts.exp_name, opts.resume)
    best_SPD, best_TC = float("inf"), 0.0

    # Load data
    if opts.model == 'vlntrans':
        opts.max_instr_len = 512
        vocab_file = "%s/vocab/vlntrans_vocab.txt" % opts.dataset
        tokenizer = tx.data.BERTTokenizer(pretrained_model_name='bert-base-uncased',
                                          hparams={'vocab_file': vocab_file})
    else:
        vocab_file = "%s/vocab/nobert_vocab.txt" % opts.dataset
        vocab = read_vocab(vocab_file)
        tokenizer = Tokenizer(vocab=vocab, encoding_length=opts.max_instr_len)
    features, img_size = load_features(opts.img_feat_dir)
    train_env = TouchdownBatch(opts, features, img_size, batch_size=opts.batch_size, seed=opts.seed,
                               splits=['train'], tokenizer=tokenizer, name="train")
    val_env = TouchdownBatch(opts, features, img_size, batch_size=opts.batch_size, seed=opts.seed,
                             splits=['dev'], tokenizer=tokenizer, name="eval")

    # Build model, optimizers, agent and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if opts.model == 'vlntrans':
        instr_encoder = tx.modules.BERTEncoder(pretrained_model_name='bert-base-uncased').to(device)
        model = VLNTransformer().to(device)
        text_linear = nn.Linear(768, opts.hidden_dim).to(device)
        pano_encoder = Conv_net(opts).to(device)

        bert_params = list(instr_encoder.parameters())
        bert_optimizer = torch.optim.Adam(bert_params, lr=opts.bert_lr, weight_decay=opts.weight_decay)
        other_params = list(pano_encoder.parameters()) + list(model.parameters()) + list(text_linear.parameters())
        optimizer = torch.optim.Adam(other_params, lr=opts.lr, weight_decay=opts.weight_decay)

        instr_encoder = nn.DataParallel(instr_encoder)
        text_linear = nn.DataParallel(text_linear)
        pano_encoder = nn.DataParallel(pano_encoder)
        model = nn.DataParallel(model)

        if opts.resume:
            model, instr_encoder, text_linear, pano_encoder, optimizer, bert_optimizer, best_SPD, best_TC = \
                resume_training(opts, model, instr_encoder, text_linear=text_linear, pano_encoder=pano_encoder,
                                optimizer=optimizer, bert_optimizer=bert_optimizer)

        agent = TouchdownVLNTransformer(opts, train_env, instr_encoder, pano_encoder, text_linear, model)
        trainer = TouchdownTrainer(opts, agent, optimizer, bert_optimizer)

    else:
        instr_encoder = Embed_RNN(len(vocab)).to(device)
        model = RConcat(opts).to(device) if opts.model == 'rconcat' else GA(opts).to(device)

        params = list(instr_encoder.parameters()) + list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=opts.lr, weight_decay=opts.weight_decay)

        if opts.resume:
            model, instr_encoder, optimizer, best_SPD, best_TC = \
                resume_training(opts, model, instr_encoder, optimizer=optimizer)

        agent = TouchdownRConcat(opts, train_env, instr_encoder, model) if opts.model == 'rconcat' else TouchdownGA(opts, train_env, instr_encoder, model)
        trainer = TouchdownTrainer(opts, agent, optimizer)
    
    # Evaluation on dev set and test set
    if opts.test:
        assert opts.resume, 'The model was not resumed.'
        test_env = TouchdownBatch(opts, features, img_size, batch_size=opts.batch_size, seed=opts.seed,
                                  splits=['test'], tokenizer=tokenizer, name='test')
        epoch = opts.start_epoch - 1
        trainer.eval_(epoch, val_env)
        trainer.eval_(epoch, test_env)
        return

    for epoch in range(opts.start_epoch, opts.max_num_epochs + 1):
        trainer.train(epoch, train_env, tb_logger)
        if epoch % opts.eval_every_epochs == 0:
            TC, SPD = trainer.eval_(epoch, val_env, tb_logger=tb_logger)
            is_best_SPD = SPD <= best_SPD
            best_SPD = min(SPD, best_SPD)
            is_best_TC = TC >= best_TC
            best_TC = max(TC, best_TC)
            print("--> Best dev SPD: {}, best dev TC: {}".format(best_SPD, best_TC))
            ckpt = {
                'opts': opts,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'instr_encoder_state_dict': instr_encoder.state_dict(),
                'best_SPD': best_SPD,
                'best_TC': best_TC,
                'optimizer': optimizer.state_dict()
            }
            if opts.model == 'vlntrans':
                ckpt['pano_encoder_state_dict'] = pano_encoder.state_dict()
                ckpt['text_linear_state_dict'] = text_linear.state_dict()
                ckpt['bert_optimizer'] = bert_optimizer.state_dict()
            save_checkpoint(ckpt, is_best_SPD, is_best_TC, epoch=epoch)
    print("--> Finished training")


if __name__ == "__main__":
    main(opts)
