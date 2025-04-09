import os
import random

test_img_dir = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\train\images"
test_img_dir = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\test\images"
annotations_dir = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\train\annotations\xmls"
txt_dir = r"C:\Users\ANSHUL M\Downloads"

all_images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(".jpg")]

num_to_keep = 500
random.seed(42)
if len(all_images) <= num_to_keep:
    print("There are fewer than or equal to 500 images. Nothing to delete.")
    chosen_images = all_images
else:
    chosen_images = random.sample(all_images, num_to_keep)

chosen_set = set(chosen_images)

chosen_list_path = os.path.join(txt_dir, "chosen_test_500_images.txt")
with open(chosen_list_path, "w") as f:
    for img_name in chosen_images:
        f.write(img_name + "\n")
print(f"Chosen images saved to {chosen_list_path}")

deleted_count = 0
for img_name in all_images:
    if img_name not in chosen_set:
        img_path = os.path.join(test_img_dir, img_name)
        if os.path.exists(img_path):
            os.remove(img_path)
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(annotations_dir, xml_name)
        if os.path.exists(xml_path):
            os.remove(xml_path)
        deleted_count += 1

print(f"Deleted {deleted_count} images and their corresponding XML annotations.")
print("Done.")
