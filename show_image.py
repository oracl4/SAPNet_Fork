#import libraries
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# create figure
fig = plt.figure(figsize=(12, 10))
  
# setting values to rows and column variables
rows = 2
columns = 2

# debug/itri/adversarial_testing/evaluations/itri_test/visualizations/frame_001540_jpg.jpg

img_name = "frame_004485_jpg.jpg"
frame_number = img_name.split("_")[1]

# img_1_path = "debug/itri/gt/evaluations/itri_test/visualizations/" + img_name
img_1_path = "debug/itri/baseline_testing/evaluations/itri_test/visualizations/" + img_name
img_2_path = "debug/itri/adversarial_testing/evaluations/itri_test/visualizations/" + img_name
img_3_path = "debug/itri/finetune_testing/evaluations/itri_test/visualizations/" + img_name
img_4_path = "debug/itri/oracle_testing/evaluations/itri_test/visualizations/" + img_name

# reading images
print(img_1_path)
Image1 = mpimg.imread(img_1_path)
Image2 = mpimg.imread(img_2_path)
Image3 = mpimg.imread(img_3_path)
Image4 = mpimg.imread(img_4_path)
  
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
fig.suptitle('Visual Comparison (Frame ' + frame_number + ")", fontsize=16)

# showing image
plt.imshow(Image1)
plt.axis('off')
plt.title("a) Baseline")
  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(Image2)
plt.axis('off')
plt.title("b) Adversarial")
  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(Image3)
plt.axis('off')
plt.title("c) Adversarial + Fine Tuning")
  
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
  
# showing image
plt.imshow(Image4)
plt.axis('off')
plt.title("d) Oracle")

plt.show()