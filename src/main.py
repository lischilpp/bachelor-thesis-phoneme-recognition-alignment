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


num_epochs = 11
batch_size = 16
initial_lr = 0.002
min_lr = 1e-10
lr_patience = 0
lr_reduce_factor = 0.4
auto_lr_find=False


def show_confusion_matrix(confmat):
    plt.figure(figsize=(15,10))

    class_names = Phoneme.folded_group_phoneme_list
    df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, cbar=False, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    dm = DataModule(batch_size)
    model = PhonemeClassifier(batch_size,
                              initial_lr,
                              min_lr,
                              lr_patience,
                              lr_reduce_factor,
                              len(DiskDataset(TRAIN_PATH)))
    trainer = pl.Trainer(gpus=1,
                         max_epochs=num_epochs,
                         auto_lr_find=auto_lr_find,
                         precision=16,
                         gradient_clip_val=0.5,
                         num_sanity_val_steps=0,
                         callbacks=[ModelCheckpoint(monitor='val_loss'),
                                    EarlyStopping(monitor='val_loss', patience=10)],
    resume_from_checkpoint='lightning_logs/version_2633/checkpoints/epoch=10-step=2419.ckpt')
    # resume_from_checkpoint='lightning_logs/version_2616/checkpoints/epoch=14-step=3299.ckpt')
    

    if auto_lr_find:
        trainer.tune(model, dm)
    else:
        trainer.fit(model, dm)
        trainer.test(datamodule=dm)
        confmat = model.confmatMetric.compute()
        show_confusion_matrix(confmat)