from settings import *
from modules.data_module import DataModule
from modules.phoneme_classifier import PhonemeClassifier
from phonemes import Phoneme

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataset.disk_dataset import DiskDataset



num_epochs = 100
batch_size = 32
initial_lr = 0.001
min_lr = 1e-8
lr_patience = 0
lr_reduce_factor = 0.4
auto_lr_find=False

if __name__ == '__main__':
    dm = DataModule(batch_size)
    dm.setup(None)
    model = PhonemeClassifier(batch_size,
                              initial_lr,
                              min_lr,
                              lr_patience,
                              lr_reduce_factor,
                              len(dm.train_dataloader()))
    trainer = pl.Trainer(gpus=1,
                         max_epochs=num_epochs,
                         auto_lr_find=auto_lr_find,
                         precision=16,
                         gradient_clip_val=0.5,
                        #  num_sanity_val_steps=0,
                         callbacks=[ModelCheckpoint(monitor='val_PER'),
                                    EarlyStopping(monitor='val_PER', patience=3)])
    # resume_from_checkpoint='lightning_logs/version_1411/checkpoints/epoch=12-step=1429.ckpt')

    if auto_lr_find:
        trainer.tune(model, dm)
    else:
        trainer.fit(model, dm)
        trainer.test(datamodule=dm)

        confmat = model.confmatMetric.compute()
        plt.figure(figsize=(15,10))

        class_names = Phoneme.folded_group_phoneme_list
        df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, cbar=False, fmt="d")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()