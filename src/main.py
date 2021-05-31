from settings import *
from modules.data_module import DataModule
from modules.phoneme_classifier import PhonemeClassifier
from phonemes import Phoneme

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


num_epochs = 100
batch_size = 32
initial_lr = 0.001
lr_patience = 1
lr_reduce_factor = 0.5
auto_lr_find=False

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(monitor='val_PER')

    dm = DataModule(batch_size)
    model = PhonemeClassifier(batch_size, initial_lr, lr_patience, lr_reduce_factor)
    trainer = pl.Trainer(gpus=1,
                         max_epochs=num_epochs,
                         auto_lr_find=auto_lr_find,
                         gradient_clip_val=0.5,
                         callbacks=[checkpoint_callback],
                         accumulate_grad_batches=2,
                         precision=16)
        # resume_from_checkpoint='lightning_logs/version_164/checkpoints/epoch=79-step=8319.ckpt')

    if auto_lr_find:
        trainer.tune(model, dm)
    else:
        trainer.fit(model, dm)
        trainer.test(datamodule=dm)

        confmat = model.confmatMetric.compute()
        plt.figure(figsize=(15,10))
        class_names = Phoneme.folded_phoneme_list
        df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()