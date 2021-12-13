import shutil
import os

  
# Define the source and destination path
for i in range(250, 260, 50):
    source = "./netG_epoch_" + str(i) + "/valid/single"
    subfolders = [ f.path for f in os.scandir(source) if f.is_dir() ]
    destination = "./netG_epoch_" + str(i) + "/valid/single"
    
    # # code to move the files from sub-folder to main folder.
    # folders = [x[0] for x in os.walk(source)]

    # files = os.listdir(source)
    for folder in subfolders:
        files = os.listdir(folder)
        for file in files:
            file_name = os.path.join(folder, file)
            shutil.move(file_name, destination)
    print("Files Moved")