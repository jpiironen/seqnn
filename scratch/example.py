import json
from logging.config import stopListening
import nltk
from numpy import block
import torch
from seqnn.data.dataset import (
    CombinationDataset,
    DictSeqDataset,
    DocumentCollection,
    TextDocument,
)
from seqnn.model.transformer_old import Transformer

path = "data/amazon_ratings/reviews_Digital_Music_5.json"

documents = []
for line in open(path, "r"):
    item = json.loads(line)
    documents.append(item["reviewText"])


documents = documents[:10]
documents = [TextDocument(d, ignore_case=True) for d in documents]
document_collection = DocumentCollection(documents)
vocab_size = len(document_collection.index_to_token)
block_size = 128

dataset = CombinationDataset(
    [
        DictSeqDataset(
            data={
                "tokens": torch.tensor(document_collection.words_to_indices(doc.words))
            },
            seq_len=block_size,
            seq_len2=0,
            split_past_future=False,
        )
        for doc in documents
    ]
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)


chunk = document_collection.indices_to_words(dataset[0]['tokens'].numpy())



#tokens = torch.tensor([0, 4, 2, 1, 1]).view(1, -1)
model = Transformer(seq_len_max=block_size, vocab_size=vocab_size, num_blocks=4)
lossfun = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []
for batch in dataloader:
    tokens = batch["tokens"]
    pred = model(tokens)

    pred = pred[:,:-1,:].contiguous().view(-1, pred.shape[-1])
    tokens = tokens[:,1:].contiguous().view(-1)
    loss = lossfun(pred, tokens)
    
    optimizer.zero_grad()
    loss.backward()
    # TODO: gradient clipping here
    optimizer.step()

    losses.append(loss.item())
    print(loss.item())
    

import matplotlib.pyplot as plt
plt.plot(losses)

print('foo')
#F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))




#out2 = model(tokens[:, :-2])
#print(out[0])
#print(out2[0])