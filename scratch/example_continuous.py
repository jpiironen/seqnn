import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seqnn
from seqnn import SeqNN, SeqNNConfig
from seqnn.utils import get_data_sample

# from seqnn.data.dataset import DictSeqDataset
# from seqnn.model.transformer_old import Transformer


np.random.seed(42)
n = 10000
time = np.arange(n)
y1 = np.sin(time / 6) + 0.1 * np.random.randn(n)
y2 = np.sin(time / 4 + 2) + 0.1 * np.random.randn(n)
y3 = np.sin(time / 3 + 1) + 0.1 * np.random.randn(n)

nvalid = 2000

df = pd.DataFrame(
    {
        "y1": torch.tensor(y1, dtype=torch.float),
        "y2": torch.tensor(y2, dtype=torch.float),
        "y3": torch.tensor(y3, dtype=torch.float),
        "t": torch.tensor(time, dtype=torch.float),
    }
)
df_train = df.iloc[:-nvalid, :]
df_valid = df.iloc[-nvalid:, :]


config = SeqNNConfig(
    targets={"obs": ["y1", "y2", "y3"]},
    #targets={"obs": ["y1", "y2"]},
    #targets={"obs": "y1"},
    #targets=["y1"],
    #controls_continuous=["y3"],
    horizon_past=20,
    horizon_future=5,
    #model="TransformerMultivariate",
    #model="TransformerUnivariate",
    #model="Transformer",
    #model_args={"num_blocks": 2},
    optimizer="SGD",
    optimizer_args={"lr": 0.001, "momentum": 0.9},
    #optimizer="Adam",
    #optimizer_args={"lr": 0.001},
    lr_scheduler_args={"gamma": 0.5, "step_size": 2000},
    #max_grad_norm=30,
)


model = SeqNN(config)


#model.save("models/testmodel")
#model2 = SeqNN.load("models/testmodel")
#for p, p2 in zip(
#    model.model.model_core.parameters(), model2.model.model_core.parameters()
#):
#    print(p - p2)
#for p, p2 in zip(
#    model.model.scaler.parameters(), model2.model.scaler.parameters()
#):
#    print(p - p2)


model.train(
    df_train, df_valid, 
    steps=15000, 
    #num_batches_scaler_train=1, 
    #num_batches_validation=1, 
    #dev_run=True
)
#model.save('models/testmodel_uv_y1y2y3_h=20_v1')
#model.save('models/testmodel_y1y2_control=y3_h=20_v1')
#model.save('models/testmodel_y1_control=y2/')


#path = 'models/testmodel_y1y2y3/'
#path = 'models/testmodel_y1_control=y2/'
#path = 'models/testmodel_nocontrol_y1y2_sequenced_v2'
#path = 'models/testmodel_y1y2y3_h=20_v1'
#model = seqnn.load(path)

#validset = model.get_dataset(df_valid, horizon_future=2)
#past, future = get_data_sample(validset, indices=0)
#out = model.predict(past, future)


#past = model.model.to_scaled(past)
#future = model.model.to_scaled(future)
#(
#    target_past,
#    control_past,
#    target_future,
#    control_future,
#) = model.model.data_handler.prepare_data(past, future, augment=True)
#model.model.model_core.get_loss(
#    target_past,
#    control_past,
#    target_future,
#    control_future,
#)

#print(out)
#print('foo')