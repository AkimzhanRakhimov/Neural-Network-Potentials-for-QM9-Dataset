drive.mount("/content/drive")
if os.path.exists("/content/drive/MyDrive/NNP"):
  print("Path exist already")
else:
  os.mkdir("/content/drive/MyDrive/NNP")
  print("Path has been created successfully")
