import PIL
import cv2

#open and dsplay image
img_path = r"F:\NGA_Similarity_Project\NGA_Images_Fixed\0032a4df-e398-4ff2-8c82-069395f276bd.jpg"

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

cv2.imshow("Image", img)
