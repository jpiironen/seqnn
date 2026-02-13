import numpy as np
import torch

from seqnn import SeqNNConfig, SeqNN
from seqnn.data.dataset import CharacterMap, DictSeqDataset

# setup model
config = SeqNNConfig(
    targets="token",
    controls=None,
    horizon_past=128,
    horizon_future=0,
    model="Transformer",
    likelihood="LikCategorical",
    likelihood_args={"num_classes": vocab_size},
    optimizer="Adam",
    optimizer_args={"lr": 0.001},
    validate_every_n_steps=5,
)
model = SeqNN(config)

# train
model.train(data_train, data_valid, overfit_batches=1, max_epochs=50)








#################################

horizon_past = 50
horizon_future = 10
dataset_train = DictSeqDataset(datatrain, seq_len=horizon_past, seq_len2=horizon_future)
dataset_valid = DictSeqDataset(datavalid, seq_len=horizon_past, seq_len2=horizon_future)
trainloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=32, shuffle=True, drop_last=True
)


# model
model = Transformer(
    seq_len_max=100,
    continuous_tokens=True,
    num_heads=1,
    dim_embed=64,
    num_blocks=1,
    layer_norm_last=True,
    dropout_attn=0.1,
    dropout_embed=0.1,
    dropout_resid=0.1,
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
training_horizon = 3
loss_train = []

# fix batch
# for step, (past, future) in enumerate(trainloader):
#    break
# for step in range(200):

for epoch in range(5):
    for step, (past, future) in enumerate(trainloader):

        # pred = model(past['y']).squeeze()[:,-1]
        pred = model.sample(past["y"], training_horizon)
        loss = ((pred - future["y"][:, :training_horizon]) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())

        print(step, len(trainloader))
        # if step == 100:
        #    break


plt.plot(loss_train)
plt.ylim(0, 0.03)

# plt.plot(time[:100], y[:100])

# plt.plot(torch.cat((past['y'], future['y']), dim=1)[0,:], '.')
# plt.plot(torch.cat((past['y'], future['y']), dim=1)[1,:], '.')


model.eval()
plt.figure()
pred = model.sample(past["y"], horizon_future)

for i, color in zip([1, 10, 20, 30], ["C0", "C1", "C2", "C3"]):
    plt.plot(pred[i, :].detach(), color=color)
    plt.plot(future["y"][i, :], ".", color=color)
    # break


plt.show()
