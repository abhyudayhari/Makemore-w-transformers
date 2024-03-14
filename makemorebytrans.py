import torch
import torch.nn.functional as F
import torch.nn as nn
import wget
import os.path

#------ HYPERPARAMETERS------
batch=64
block=8
n_heads=4
n_layers=10
n_embed=384
lr=3e-4
iters=10000
eval_iter=500
dropout=0.2
eval_iters=500

#------------------------------



if os.path.isfile('names.txt'):
   print("found file")
else:
    wget.download("https://raw.githubusercontent.com/karpathy/makemore/master/names.txt")
words = open('names.txt', 'r').read().splitlines()
a="\n".join(words)
ch=sorted(list(set("".join(a))))
itoch ={i:c for i,c in enumerate(ch)}
chtoi={c:i for i,c in itoch.items()}
vocab_size=len(ch)
encode=lambda s: [chtoi[char] for char in s]
decode=lambda i: "".join([itoch[ix] for ix in i])
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_batches(split):
  n=int(0.9*len(a))
  if split=="train":
    data=torch.tensor(encode(a[:n]))
  else:
    data= torch.tensor(encode(a[n:]))
  ix=torch.randint(len(data)-block,(batch,))
  x=torch.stack([data[i:i+block] for i in ix])
  y=torch.stack([data[i+1:i+block+1] for i in ix])
  x,y=x.to(device),y.to(device)
  return x,y
class Head(nn.Module):
  def __init__(self,head):

    super().__init__()

    self.query=nn.Linear(n_embed,head,bias=False)
    self.key=nn.Linear(n_embed,head,bias=False)
    self.value=nn.Linear(n_embed,head,bias=False)
    self.dropout=nn.Dropout(dropout)
    self.register_buffer('tril',torch.tril(torch.ones(block,block)))
  def __call__(self,x):
    B,T,C=x.shape
    query=self.query(x)
    key=self.key(x)
    value=self.value(x)

    wei=query @ key.transpose(-2,-1)*C**-0.5
    wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
    wei=F.softmax(wei,dim=-1)
    wei=self.dropout(wei)
    wei=wei@value
    return wei
class MultiHead(nn.Module):
    def __init__(self,n_embeds,n_heads):
      super().__init__()
      self.multihead=nn.ModuleList(Head(head=n_embeds) for _ in range(n_heads))
    def __call__(self,x):
      #out=torch.cat([h(x) for h in self.multihead])

      out=torch.cat([h(x) for h in self.multihead],dim=-1)
      return out
class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()
    self.net=nn.Sequential(nn.Linear(n_embed,n_embed*4),
    nn.ReLU(),nn.Linear(n_embed*4,n_embed),nn.Dropout(dropout))
  def __call__(self,x):
    out=self.net(x)
    return out
class Block(nn.Module):
  def __init__(self):
    super().__init__()
    n_embeds=n_embed//n_heads
    self.ln1=nn.LayerNorm(n_embed)
    self.mask1=MultiHead(n_embeds,n_heads=n_heads)
    self.ln2=nn.LayerNorm(n_embed)
    self.mask2=MultiHead(n_embeds,n_heads=n_heads)
    self.ln3=nn.LayerNorm(n_embed)
    self.ffd=FeedForward()
    self.ln3=nn.LayerNorm(n_embed)
    self.ln4=nn.LayerNorm(n_embed)
  def __call__(self,x):
    x=x+self.mask1(self.ln1(x))
    x=x+self.mask2(self.ln2(x))
    x=x+self.ffd(self.ln3(x))
    x=x+self.ln4(x)
    return x
class Model(nn.Module):
  def __init__(self):
    super().__init__()


    self.embedding=nn.Embedding(vocab_size,n_embed) #B,T,C
    self.sahead=nn.Sequential(*[Block() for _ in range(n_layers)])
    self.lin=nn.Linear(n_embed,vocab_size)
    weight=torch.ones(vocab_size,n_embed)
    bias=torch.ones(vocab_size)
   
    self.posembed=nn.Embedding(block,n_embed)
    with torch.no_grad():
      self.lin.weight.copy_(weight)
      self.lin.bias.copy_(bias)
  def __call__(self,x,target=None):
    B,T=x.shape
    posembed=self.posembed(torch.arange(T,device=device))

    x=self.embedding(x)+posembed
    x=self.sahead(x)
    logits=self.lin(x)
    #x=F.softmax(x,dim=-1)
    #return x
    if target==None:
      loss=None
    else:
      B,T,C=x.shape
      loss=F.cross_entropy(logits.view(B*T,-1),target.view(-1))
    return loss,logits
  def generate(self,ix,n):
    count=0
    while count!=n:
      
      if ix[:,-1]==0:
        count+=1

      ix_cond=ix[:,-block:]
      loss,logits=self(ix_cond)
      logits=logits[:,-1,:]
      probs=F.softmax(logits,dim=-1)
      idx_next=torch.multinomial(probs,num_samples=1)
      ix=torch.cat((ix,idx_next),dim=1)
    return ix
m=Model()
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batches(split)
            loss,logits= m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out
m.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

def training():
  print("running on ",device)


  for steps in range(iters): # increase number of steps for good results...

      # sample a batch of data
      xb, yb = get_batches('train')
      xb.to(device)
      yb.to(device)

      #evaluate the loss
      loss,logits = m(xb, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      # print(loss.item())
      # print(loss.item())
      # break
      if steps % eval_iters == 0 or iter == iters - 1:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
training()
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long,device=device), n=10)[0].tolist()))
