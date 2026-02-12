for epoch in range(50):
  model.train()
  total_loss=0
  for batch in tqdm(train_dataloader):
    z=batch['z'].to(device)
    pos=batch['pos'].to(device)
    energy=batch['energy'].to(device)
    # forward pass

    energy_pred=model(z,pos)
    # count the loss
    loss=loss_fn(energy_pred,energy)
    # zero grad
    optimizer.zero_grad()
    # backprop
    loss.backward()
    # update the weights
    optimizer.step()
    total_loss+=loss
  print(f"Loss:{total_loss/len(train_dataloader)}")
