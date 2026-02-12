device="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class AtomMLP(nn.Module):
  def __init__(self,n_atom_types=100,hidden_dim=64,output_features=1):
    super().__init__()
    self.embeddings=nn.Embedding(n_atom_types,embedding_dim=64)
    # Pytorch architecture allows us to use bathes(atoms in molecula) in neural network loop
    self.mlp=nn.Sequential(
        nn.Linear(hidden_dim+15,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,1)
    )
  def forward(self,z,pos):


    dists=torch.cdist(pos,pos)
    N=pos.shape[1]

    mask=1-torch.eye(N,device=pos.device)
    dists=dists*mask
    dists_sorted,_=torch.sort(dists,dim=2)
    if N<=15:
      zeros=torch.zeros((1,N,16-N),device=pos.device)
      dists_sorted=torch.cat([dists_sorted,zeros],dim=2)

    dists_sorted=dists_sorted[:,:,1:16]

    h=self.embeddings(z.long())
    atom_input=torch.cat([h,dists_sorted],dim=2)
    energy=self.mlp(atom_input).sum()
    return energy

# Here i create GNN architecture

class AtomEmbeddings(nn.Module):
  def __init__(self,num_types,hidden_dim):
    super().__init__()
    self.embedding=nn.Embedding(num_types,hidden_dim)

  def forward(self,z):
    h=self.embedding(z.long())
    return h

class MessagePassingLayerV0(nn.Module):
  def __init__(self,hidden_dim):
    super().__init__()
    self.mlp=nn.Sequential(
        nn.Linear(2*hidden_dim+1,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,hidden_dim)
    )
  def forward(self,h,edge_index,edge_attr):
    m=torch.zeros_like(h).to(device)
    for idx in range(edge_index.shape[1]):
      i,j=edge_index[:,idx]
      msg_input=torch.cat([h[i],h[j],edge_attr[idx]],dim=0)
      m[i]+=self.mlp(msg_input)
    h=h+m
    return h
class MessagePassingLayerV1(nn.Module):
  def __init__(self,hidden_dim):
    super().__init__()
    self.mlp=nn.Sequential(
        nn.Linear(hidden_dim+1,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,hidden_dim)
    )
  def forward(self,h,edge_index,edge_attr):
    m=torch.zeros_like(h)
    for idx in range(edge_index.shape[1]):
      i,j=edge_index[:,idx]
      msg_input=torch.cat([h[j],edge_attr[idx]])
      m[i]+=self.mlp(msg_input)
    h=h+m
    return h
class EnergyHead(nn.Module):
  def __init__(self,hidden_dim):
    super().__init__()
    self.mlp=nn.Sequential(
        nn.Linear(hidden_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,1)
    )
  def forward(self,h):
    return self.mlp(h).sum()
class AtomGNN(nn.Module):
  def __init__(self,num_types,hidden_dim,layers_num=3):
    super().__init__()
    self.embedding=AtomEmbeddings(num_types,hidden_dim)
    self.embedding.to(device)
    self.layers=nn.ModuleList([MessagePassingLayerV0(hidden_dim) for _ in range(layers_num)])
    for layer in self.layers:
      layer.to(device)
    self.energy=EnergyHead(hidden_dim)
    self.energy.to(device)
  def forward(self,z,pos,cutoff=5.0):
    N=pos.shape[1]
    dists=torch.cdist(pos,pos).squeeze(0)
    h=self.embedding(z).squeeze(0)
    mask=(dists<cutoff) & (~torch.eye(N, dtype=bool,device=device ))
    edge_index=mask.nonzero(as_tuple=False).T
    edge_attr=dists[mask].unsqueeze(1)
    for layer in self.layers:
      h=layer(h,edge_index,edge_attr)
    return self.energy(h)
