import os, pathlib, random, shutil

ROOT_FOLDER = "dataset/26_05_12_Truck_Pose"
IMG_FOLDER = "images"
LAB_FOLDER = "labels"

IMG_DIR = os.path.join(ROOT_FOLDER, IMG_FOLDER)
LAB_DIR = os.path.join(ROOT_FOLDER, LAB_FOLDER)
IMG_TRAIN_DIR = os.path.join(IMG_DIR, "train")
IMG_VAL_DIR = os.path.join(IMG_DIR, "val")
LAB_TRAIN_DIR = os.path.join(LAB_DIR, "train")
LAB_VAL_DIR = os.path.join(LAB_DIR, "val")

os.makedirs(os.path.join(IMG_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(IMG_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(LAB_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(LAB_DIR, "val"), exist_ok=True)

TRAIN_RATIO = 0.8

def move_file(src_dict, dst_dict):
  lst_files = [file for file in os.scandir(src_dict) if file.is_file()]
  idx_split = int(len(lst_files) * TRAIN_RATIO)

  train_dataset = lst_files[:idx_split]
  val_dataset = lst_files[idx_split:]

  print(f"Training on {len(train_dataset)} images and validating on {len(val_dataset)} images")

  for idx, file in enumerate(lst_files):
    basename = os.path.basename(file)
    filename = basename.split('.')[0]
    
    if idx > idx_split:
      shutil.move(file, dst_dict)

if __name__ == "__main__":
  move_file(IMG_TRAIN_DIR, IMG_VAL_DIR)
  move_file(LAB_TRAIN_DIR, LAB_VAL_DIR)


  
    