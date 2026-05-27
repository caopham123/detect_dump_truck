import os, pathlib, random, shutil

ROOT_FOLDER = os.path.join(os.path.dirname(__file__), "dataset", "26_05_23_dump_detect_task")
IMG_TRAIN_DIR = os.path.join(ROOT_FOLDER, "images", "train")
IMG_VAL_DIR = os.path.join(ROOT_FOLDER, "images", "val")
LAB_TRAIN_DIR = os.path.join(ROOT_FOLDER, "labels", "train")
LAB_VAL_DIR = os.path.join(ROOT_FOLDER, "labels", "val")

# Create val folders
os.makedirs(IMG_VAL_DIR, exist_ok=True)
os.makedirs(LAB_VAL_DIR, exist_ok=True)

TRAIN_RATIO = 0.8

def split_dataset():
  print(f"ROOT_FOLDER: {ROOT_FOLDER}")
  print(f"IMG_TRAIN_DIR: {IMG_TRAIN_DIR}")
  
  # Get all image filenames in the training directory
  lst_files = [file.name for file in os.scandir(IMG_TRAIN_DIR) if file.name.endswith(".jpg")]
  print(f"Found {len(lst_files)} images in train folder!")
  
  if len(lst_files) == 0:
    print("No files found to split. Make sure images are in the train folder.")
    return

  # Random shuffle works in-place and returns None, so we just call it
  random.shuffle(lst_files)
  
  # Split into train and val
  idx_split = int(len(lst_files) * TRAIN_RATIO)
  train_dataset = lst_files[:idx_split]
  val_dataset = lst_files[idx_split:]

  print(f"Keeping {len(train_dataset)} for training, moving {len(val_dataset)} to validation...")

  for file_name in val_dataset:
    # Move image
    src_img = os.path.join(IMG_TRAIN_DIR, file_name)
    dst_img = os.path.join(IMG_VAL_DIR, file_name)
    shutil.move(src_img, dst_img)
    
    # Move corresponding label
    label_name = os.path.splitext(file_name)[0] + ".txt"
    src_lab = os.path.join(LAB_TRAIN_DIR, label_name)
    dst_lab = os.path.join(LAB_VAL_DIR, label_name)
    
    if os.path.exists(src_lab):
      shutil.move(src_lab, dst_lab)
      
  print("Successfully split the dataset!")

if __name__ == "__main__":
  split_dataset()