import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os           #for importing files by all filenames
from PIL import Image

def ConvertGreekImages(folder):

    f1 = open('GreekPixels.csv', 'w')
    f1.write("Greek Letter Intensity \n")

    f2 = open('GreekLabel.csv', 'w')
    f2.write('Greek Label \n')

    for filename in os.listdir(folder):

        img = cv.imread(os.path.join(folder, filename), 0) #!! using imread has caused image becomes 3 channels!!

        if img is not None:
            
            ret, img_T = cv.threshold(img, 180, 255, cv.THRESH_BINARY_INV)
            img_T = cv.resize(img_T, (28, 28))
            img_T = cv.cvtColor(img_T, cv.COLOR_BGR2RGB)

            print (filename)
            plt.imshow(img_T)
            plt.show()

            for r in range(28):
                for c in range(28):
                    pixel = str(img_T[r][c][0]) + ", "
                    f1.write(pixel)
            f1.write("\n")

            if filename.split("_")[0] == "alpha":
                f2.write("0 \n")
            elif  filename.split("_")[0] == "beta":
                f2.write("1 \n")
            elif  filename.split("_")[0] == "gamma":
                f2.write("2 \n")

    f1.close()
    f2.close()

    return

def main(argv):
    ConvertGreekImages('greek-1')
    #cv.imwrite('CascaHandWritting/formatted/9_binary.jpg', cv.cvtColor(img_T, cv.COLOR_BGR2GRAY))
    return

if __name__ == "__main__":
    main(sys.argv)
