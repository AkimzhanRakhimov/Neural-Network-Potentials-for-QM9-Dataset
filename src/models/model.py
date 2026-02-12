# create a model
model=AtomMLP()
# model=AtomGNN(10,64,3)
model.to(device)

train_dataloader=DataLoader(dataset=training_dataset,batch_size=1,shuffle=True)

optimizer=torch.optim.Adam(params=model.parameters(),lr=1e-2)
loss_fn=nn.MSELoss()
