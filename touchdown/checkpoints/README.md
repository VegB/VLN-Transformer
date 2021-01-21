## Pre-trained Checkpoints

### Download
Here we provide the pre-trained checkpoints for RCONCAT, GA and VLN Transformer used in our experiment.
Please download the checkpoints with the script:

```bash
cd checkpoints  # checkpoints should be placed in VLN-Transformer/touchdown/checkpoints
python download.py
```

### Checkpoints
We provide five set of checkpoints for RCONCAT and VLN Transformer, which should be place at ```checkpoints/rconcat/``` and ```checkpoints/vlntrans/``` after running the download script.

- ```vanilla```: Navigation agent trained on ```touchdown``` dataset without pre-training on auxiliary datasets.
- ```finetuned_manh50```: Pre-trained on ```manh50``` dataset, and finetuned on ```touchdown``` dataset.
- ```finetuned_mask```: Pre-trained on ```manh50_mask``` dataset, and finetuned on ```touchdown``` dataset.