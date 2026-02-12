torch.save(obj=model.state_dict(),f="/content/drive/MyDrive/NNP/model_1.pth")
print("Model has been saved")

model.load_state_dict(torch.load("/content/drive/MyDrive/NNP/model_1.pth"))
print("Model has been loaded")
print(model.state_dict())
