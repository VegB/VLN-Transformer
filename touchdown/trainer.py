import torch
from utils import load_datasets, AverageMeter
import time
import math


class TouchdownTrainer:
    def __init__(self, opts, agent, optimizer, bert_optimizer=None):
        self.opts = opts
        self.agent = agent
        self.optimizer = optimizer
        self.bert_optimizer = bert_optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, train_env, tb_logger=None):
        print('Training on {} env ...'.format(train_env.splits[0]))
        self.agent.env = train_env
        self.agent.model.train()
        self.agent.instr_encoder.train()
        if self.opts.model == 'vlntrans':
            self.agent.text_linear.train()
            self.agent.pano_encoder.train()
        self.agent.env.reset_epoch()

        losses = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()
        self.train_iters_epoch = math.ceil(len(train_env.data) / self.opts.batch_size)
        for iter_ in range(1, self.train_iters_epoch + 1):
            loss, _ = self.agent.rollout(is_test=False)
            self.optimizer.zero_grad()
            if self.opts.finetune_bert:
                self.bert_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.opts.finetune_bert:
                self.bert_optimizer.step()
            batch_time.update(time.time() - end)
            losses.update(loss.item(), len(self.agent.env.batch))
            end = time.time()

            if tb_logger and iter_ % 10 == 0:
                current_iter = iter_ + (epoch - 1) * self.train_iters_epoch
                tb_logger.add_scalar('train/loss_train', loss, current_iter)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(
                epoch, iter_, self.train_iters_epoch, batch_time=batch_time,
                loss=losses), end='')
        if tb_logger:
            tb_logger.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            if self.opts.finetune_bert:
                tb_logger.add_scalar('epoch/bert_learning_rate', self.bert_optimizer.param_groups[0]['lr'], epoch)
            tb_logger.add_scalar('epoch/train/loss', losses.avg, epoch)

    def eval_(self, epoch, val_env, tb_logger=None):
        phase = val_env.env.name
        print('Evaluating on {} env ...'.format(phase))

        losses = AverageMeter()
        batch_time = AverageMeter()

        self.agent.env = val_env
        self.agent.env.reset_epoch()
        self.agent.model.eval()
        self.agent.instr_encoder.eval()
        if self.opts.model == 'vlntrans':
            self.agent.text_linear.eval()
            self.agent.pano_encoder.eval()

        val_iters_epoch = math.ceil(len(val_env.data) / self.opts.batch_size)

        metrics = [0] * 3  # [TC, SPD, SED]
        if self.opts.CLS:
            metrics += [0]
        if self.opts.DTW:
            metrics += [0] * 5
        with torch.no_grad():
            end = time.time()
            for iter_ in range(1, val_iters_epoch + 1):
                _, trajs = self.agent.rollout(is_test=True)
                self.agent.env.eva_metrics(trajs, metrics)

                batch_time.update(time.time() - end)
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    epoch, iter_, val_iters_epoch, batch_time=batch_time))

        metrics = [m / len(val_env.data) for m in metrics]
        metrics = [m * 100 if m < 1 else m for m in metrics]
        if tb_logger:
            tb_logger.add_scalar('epoch/{}/TC'.format(phase), metrics[0], epoch)
            tb_logger.add_scalar('epoch/{}/SPD'.format(phase), metrics[1], epoch)
            tb_logger.add_scalar('epoch/{}/SED'.format(phase), metrics[2], epoch)

        print("=======[%s] Evaluation Metrics=======" % phase)
        print("TC: %.2f, SPD: %.2f, SED: %.2f" % tuple(metrics[:3]), end='')
        if self.opts.CLS:
            print(', CLS:%.2f' % metrics[3], end='')
        if self.opts.DTW:
            print(', DTW:%.2f, nDTW:%.2f, SDTW:%.2f' % tuple(metrics[-3:]))
        else:
            print('')
        print("================================")

        return metrics[0], metrics[1]
