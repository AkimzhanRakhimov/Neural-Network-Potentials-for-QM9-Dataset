dataset=QM9(root="qm9_data")

class Bounded_Dataset(Dataset):
  def __init__(self,dataset_name,n_samples=10000):
    self.data=dataset_name[:n_samples]

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    d=self.data[idx]
    return{
        "z":d.z.float(),
        "pos":d.pos.float(),
        "energy":d.y[0,10].float()
    }
training_dataset=Bounded_Dataset(dataset)
