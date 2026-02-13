import numpy as np
import pandas as pd
import torch
import seqnn
from seqnn.utils import get_data_sample


path = 'models/testmodel/'
model = seqnn.load(path)

n = 10000
time = np.arange(n)
y = np.sin(time / 6) + 0.01 * np.random.randn(n)

nvalid = 2000

df = pd.DataFrame(
    {
        "y": torch.tensor(y[:-nvalid], dtype=torch.float),
        "t": torch.tensor(time[:-nvalid], dtype=torch.float),
    }
)
df_train = df.iloc[:-nvalid, :]
df_valid = df.iloc[-nvalid:, :]



validset = model.get_dataset(df_valid)

past, future = get_data_sample(validset, indices=0)

pred = model.predict(past, future)
print(pred)