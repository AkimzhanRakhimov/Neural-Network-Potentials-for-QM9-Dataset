def get_masses(all_masses):
  masses=[atomic_masses[mass] for mass in all_masses]
  return torch.tensor(masses).unsqueeze(dim=1)
atomic_masses={1:1.0,6:12.0,7:14.0,8:16.0,9:19.0}
batch=next(iter(train_dataloader))
z=batch["z"]
pos=batch["pos"]
pos.requires_grad=True
z_list=list(map(int,z.squeeze(dim=0).tolist()))
masses=get_masses(z_list)
atoms=Atoms(numbers=z_list,positions=pos.squeeze().detach().numpy())
traj=[]
print(atoms.positions)
def integrator(model,masses,z,pos,u,dt=0.002):
  pos.requires_grad_=True
  energy_pred=model(z,pos)

  forces=-torch.autograd.grad(energy_pred,pos,create_graph=True)[0].squeeze(dim=0)
  acc=forces/masses
  u_new=u+0.5*acc*dt

  pos_new=pos+u_new*dt
  return pos_new,u_new
pos_new=pos
u_new=0

