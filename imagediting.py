# install libraries usin pip command
import cv2
import matplotlib.pyplot as plt
import numpy as np


    #IMAGE AND RESIZED IMAGES
img = cv2.imread("1.jpg")
resized = cv2.resize(img,(300,300))

img1 = cv2.imread("2.jpg")
resized1 = cv2.resize(img1,(300,300))

   #GRAYSCALING
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

   #NEGATIVE
negative = 1 - resized

   #CREATING NEW IMAGES
img = np.zeros((300,500, 3),dtype=np.uint8)
img[:]=[255,200,100]

    #ADD TEXT
text=cv2.putText(img, "The image", (100,100), cv2.FONT_ITALIC, 2, (0,0,200))

  #method for splitting
b=resized[:,:,0]
g=resized[:,:,1]
r=resized[:,:,2]

   # for splitting but its costly method
b,g,r = cv2.split(resized)
rgb = cv2.merge([r,g,b])

    # to convert
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

  #CONCATENATE
horizontal = cv2.hconcat([resized, resized1])
vertical = cv2.vconcat([resized, resized1])

cv2.imshow('image1', horizontal)
cv2.imshow('image2', vertical)

    #CROPPED
cropped = resized1[116:176,140:250]
r = cv2.selectROI(resized1, False)
cropped = resized1[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]

    #ROTATE
angle =cv2.ROTATE_90_CLOCKWISE
rotated = cv2.rotate(resized1, angle)

    #FLIP
flipped = cv2.flip(resized1, 0)  #or 1


    #BORDER
bordered = cv2.copyMakeBorder(resized1, 10,10,10,10, cv2.BORDER_CONSTANT, value=(0,50,90))


   # FILTERS

   #SATURATION
imghsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype('float32')
h,s,v = cv2.split(imghsv)
s=s * 3
s = np.clip(s, 0, 255)
imghsv = cv2.merge([h,s,v])
saturated = cv2.cvtColor(imghsv.astype('uint8'), cv2.COLOR_HSV2BGR)

     #BLUR
avgBlur = cv2.blur(resized1, (3,3))
gausBlur = cv2.blur(resized1, (3,3), 0)


      #SMOOTH
smooth = cv2.edgePreservingFilter(resized, cv2.RECURS_FILTER, 60, 1)


    #PENCIL SHADE
pencil, colored = cv2.pencilSketch(resized, 200, 0.1, shade_factor=0.1)



       #CONTRAST AND BRIGHTNESS
alpha = 1
beta = 5
adjusted = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)

     #SHARPEN
kernel = np.array([[0, -1, 0],
         [1, 4, -1],
         [0, -1, 0]])
sharpness = cv2.filter2D(resized, -1, kernel)



   #SHOW

cv2.imshow("resized", resized)
cv2.imshow("resized", resized1)

cv2.imshow('original', resized)

cv2.imshow("grayscaled", gray)

cv2.imshow("negative",negative)

cv2.imshow("blank", img)

cv2.imshow("text",text)

plt.imshow(rgb)
plt.show(rgb)

cv2.imshow('rotated', rotated)

cv2.imshow('flipped', flipped)

cv2.imshow('bordered', bordered)

cv2.imshow('saturated', saturated)



cv2.imshow('blur', avgBlur)
cv2.imshow('gblur', gausBlur)

cv2.imshow('smooth', smooth)

cv2.imshow('pencil', pencil)
cv2.imshow('c-sketch', colored)

cv2.imshow('adjusted',adjusted )




   #SAVE IMAGE
cv2.imwrite("10.png", img)


   #END
cv2.waitKey()
cv2.destroyAllWindows()