import cv2

img = cv2.imread('OrigImage.jpg')
flippedImage = cv2.flip(img,1)
cv2.imwrite('ImageRef/flippedImage.jpg', flippedImage)
