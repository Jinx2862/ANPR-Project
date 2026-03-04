# pyre-ignore[21]
import cv2

img = cv2.imread("sample.jpg")

# Create a window that can be resized
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)


#resized_img = cv2.resize(img, (700, 700))

# Resize the window to 600x600 (the image will scale to fit)
#cv2.resizeWindow("Image", 600, 600)

cv2.rectangle(img,(300,300),(900,900),(255,0,0),3)

cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
