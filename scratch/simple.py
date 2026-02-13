import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from seqnn.data.dataset import DictSeqDataset
from seqnn.config import SeqNNConfig
from seqnn.model.seqmodel import SeqNN


config = SeqNNConfig(
    targets='y',
    controls=None,
    horizon_past=10,
    horizon_future=5,
    optimizer_args={"lr": 0.0001, "momentum": 0.9},
)
model = SeqNN(config)


n = 10000
time = np.arange(n)
y = np.sin(time / 6) + 0.01 * np.random.randn(n)
y = y.reshape(n,1)
x = np.zeros((n, 0))

#plt.plot(y[:30])
#plt.show()

datatrain = {
    "y": torch.tensor(y, dtype=torch.float),
    "x": torch.tensor(x, dtype=torch.float),
}
dataset = DictSeqDataset(datatrain, seq_len=10, seq_len2=5)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)


trainer = pl.Trainer(
    max_epochs=10,
    log_every_n_steps=1, 
    #check_val_every_n_epoch=10, 
    #callbacks=[LitProgressBar()]
)
trainer.fit(model, trainloader)#, val_dataloaders=trainloader)

#for past, future in dataloader:
#    break

#out = model.model_core(past['y'], past['x'], future['x'])
#print(out.shape)