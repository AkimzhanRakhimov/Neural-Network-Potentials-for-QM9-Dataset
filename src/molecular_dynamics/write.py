for step in range(500):
  traj.append(atoms.copy())
  pos_new,u_new=integrator(model,masses,z,pos_new,u_new,dt=0.02)
  atoms=Atoms(numbers=z_list,positions=pos_new.squeeze().cpu().clone().detach().numpy())

write("/content/drive/MyDrive/NNP/Methan_dynamic.xyz",traj)
