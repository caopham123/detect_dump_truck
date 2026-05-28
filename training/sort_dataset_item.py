import os, cv2, shutil


ROOT_DATASET = os.path.join(os.path.dirname(__file__), "dataset", "26_05_23_dump_detect_task")
IMG_FOLDER = "images"
LAB_FOLDER = "labels"
IMG_FOLDER_NEW = "sorted_images"
LAB_FOLDER_NEW = "sorted_labels"
IMG_TRAIN_FOLDER = "train"
IMG_VAL_FOLDER = "val"


def sort_dataset():
  img_folder_path = os.path.join(ROOT_DATASET, IMG_FOLDER)
  lab_folder_path = os.path.join(ROOT_DATASET, LAB_FOLDER)

  img2_folder_path = os.path.join(ROOT_DATASET, IMG_FOLDER_NEW)
  lab2_folder_path = os.path.join(ROOT_DATASET, LAB_FOLDER_NEW)
  if not os.path.exists(img2_folder_path):
    os.makedirs(img2_folder_path)
  if not os.path.exists(lab2_folder_path):
    os.makedirs(lab2_folder_path)

  img_train_path = os.path.join(img_folder_path, IMG_TRAIN_FOLDER)
  img_val_path = os.path.join(img_folder_path, IMG_VAL_FOLDER)

  img2_train_path = os.path.join(img2_folder_path, IMG_TRAIN_FOLDER)
  img2_val_path = os.path.join(img2_folder_path, IMG_VAL_FOLDER)
  if not os.path.exists(img2_train_path):
    os.makedirs(img2_train_path)
  if not os.path.exists(img2_val_path):
    os.makedirs(img2_val_path)

  lab_train_path = os.path.join(lab_folder_path, IMG_TRAIN_FOLDER)
  lab_val_path = os.path.join(lab_folder_path, IMG_VAL_FOLDER)

  lab2_train_path = os.path.join(lab2_folder_path, IMG_TRAIN_FOLDER)
  lab2_val_path = os.path.join(lab2_folder_path, IMG_VAL_FOLDER)
  if not os.path.exists(lab2_train_path):
    os.makedirs(lab2_train_path)
  if not os.path.exists(lab2_val_path):
    os.makedirs(lab2_val_path)

  files_img = [file for file in os.listdir(img_train_path) if file.endswith(".jpg")]
  files_lab = [file for file in os.listdir(lab_train_path) if file.endswith(".txt")]
  files_img_val = [file for file in os.listdir(img_val_path) if file.endswith(".jpg")]
  files_lab_val = [file for file in os.listdir(lab_val_path) if file.endswith(".txt")]

  files_img.sort()
  files_lab.sort()
  files_img_val.sort()
  files_lab_val.sort()
  print(f"files_img: {len(files_img)}")
  print(f"files_lab: {len(files_lab)}")
  print(f"files_img_val: {len(files_img_val)}")
  print(f"files_lab_val: {len(files_lab_val)}")

  print(f"img_train_path: {img_train_path}")
  for idx, file in enumerate(files_img):
    basename = os.path.basename(file)
    filename = basename.split(".jpg")[0]
    if idx < 5:
      print(f"filename: {filename}")
    shutil.copy(os.path.join(img_train_path, file), os.path.join(img2_train_path, file))
    if os.path.exists(os.path.join(lab_train_path, f"{filename}.txt")):
      shutil.copy(os.path.join(lab_train_path, f"{filename}.txt"), os.path.join(lab2_train_path, f"{filename}.txt"))

  print(f"img_val_path: {img_val_path}")
  for idx, file in enumerate(files_img_val):
    basename = os.path.basename(file)
    filename = basename.split(".jpg")[0]
    if idx < 5:
      print(f"filename: {filename}")
    shutil.copy(os.path.join(img_val_path, file), os.path.join(img2_val_path, file))
    if os.path.exists(os.path.join(lab_val_path, f"{filename}.txt")):
      shutil.copy(os.path.join(lab_val_path, f"{filename}.txt"), os.path.join(lab2_val_path, f"{filename}.txt"))

if __name__ == '__main__':
  print("begining............")
  sort_dataset()
  print("end............")