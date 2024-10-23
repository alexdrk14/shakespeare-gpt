from random import weibullvariate

import torch, os, wget
import torch.nn as nn
from torch.nn import functional as F
########################################################################################################################
###model hyperparameters
########################################################################################################################
batch_size = 64         #number of sequences to be process in parallel
block_size = 256        #aka context maximum length of the sequence
max_iters = 5000        #number of training iterations
eval_interval = 500     #number of iterations between the evaluation of the model
eval_iters = 200        #number of the batches used for the model loss evaluation
learning_rate = 3e-4    #the learning rate of the model
n_embed = 384           #number of the dimension of the embeddings
n_layers = 6            #number of sequential self-attention and feed forward blocks
n_heads = 6             #number of the head in each layer
dropout = 0.2           #drop out percentage
device = 'cuda' if torch.cuda.is_available() else 'cpu' #select gpu if it's available on device

#If input file not exist download the input txt
if not os.path.exists('input.txt'):
    _ = wget.download('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#count number of unique characters in the document in order to measure vocabulary size
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#Split dataset in 90/10 ratio for train and test portions
data = torch.tensor(encode(text), dtype=torch.long)
# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train dataset and the rest is validation
train_data = data[:n]
val_data = data[n:]

#get data in the form of the batch
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    #before return the data splits, we also move them into the device
    return x.to(device), y.to(device)

@torch.no_grad()#ignore the function during the backpropagation
def estimate_loss():
    out = {}
    model.eval() #currently not used, but in general allow to switch model of the model to eval phase and ingore the model update

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xbl, ybl = get_batch(split) # xb,yb batch samples where l stand for loss, since we use them in the estimate_loss function
            logits, loss = model(xbl, ybl)
            losses[k] = loss.item()
        out[split] = losses.mean() #compute the mean loss of the train and val split

    model.train() #currently not used, but define the switch of the model back to train mode
    return out

def training_loop(model):
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        # evaluate the model every eval_interval step
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f'#step:{iter} train loss:{losses["train"]:.4f} val loss: {losses["val"]:.4f}')

        # get new batch of the train data samples
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        # back propagate the loss and make optimization step
        loss.backward()
        optimizer.step()

class Head(nn.Module):
    """single head of the self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T, head_size)
        q = self.query(x)  # (B, T, head_size)

        #compose attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C **-0.5 # (B, T, head_size) @ (B, head_size, T) => (B, T, T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, C)
        return wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

class MultiHeadAttention(nn.Module):
    """multiple heads of the self-attention mechanism in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, head_size * num_heads)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.ffnet = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.ffnet(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)

        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))
        return x


class CharacterBasedGPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)



    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embed = self.token_embedding_table(idx) #(B,T,n_embed) == (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embed + pos_embed #(B, T,C)
        x = self.blocks(x) # Apply sequential computational blocks of self-attention and Feed Forward
        x = self.layer_norm(x) # Normalize the X before the linear head
        logits = self.lm_head(x) #(B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop input to the block_size input
            idx_cond = idx[:, -block_size:] if idx.shape[1] >= block_size else idx
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = CharacterBasedGPT(vocab_size)
model = model.to(device)

if not os.path.exists('shakespeare_gpt_model.pt'):
    training_loop(model)
    # Save model parameters after the training
    torch.save(model, 'shakespeare_gpt_model.pt')
else:
    #load trained model from file
    model = torch.load('shakespeare_gpt_model.pt', weights_only=False)
model.eval()

#Create empty context to complete and get the model generation predictions parsed through the decode function
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))