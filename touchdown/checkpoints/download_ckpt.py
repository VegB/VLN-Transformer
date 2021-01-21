"""
Download RCONCAT and VLN Transformer checkpoints.
"""
import os
import texar.torch as tx

models = ['rconcat', 'vlntrans', 'ga']
experiments = ['vanilla', 'finetuned_manh50', 'finetuned_mask']
model_name = {'rconcat': 'RCONCAT', 'vlntrans': 'VLN Transformer', 'ga': 'GA'}


def check_dir(model):
    if not os.path.isdir(model):
        os.mkdir(model)
        print('Created directory %s/.' % model)
    else:
        print('Directory %s/ exists.' % model)


def download_ckpt(model, exp):
    ckpt_url = 'https://vln-transformer-ckpt.s3-us-west-1.amazonaws.com/%s/%s.tar.gz' % (model, exp)
    ckpt_dir = './%s/%s' % (model, exp)
    if not os.path.exists(ckpt_dir):
        tx.data.maybe_download(urls=ckpt_url, path='./%s' % model, extract=True)
        os.remove('./%s/%s.tar.gz' % (model, exp))
        print('Downloaded %s checkpoint: %s, stored at %s' % (model_name[model], exp, ckpt_dir))
    else:
        print('%s/ already exists! Please rename or remove this '
              'checkpoint directory to avoid overwrite.' % ckpt_dir)


if __name__ == '__main__':
    for model in models:
        check_dir(model)
        for exp in experiments:
            download_ckpt(model, exp)
