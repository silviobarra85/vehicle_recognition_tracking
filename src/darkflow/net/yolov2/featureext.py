import cv2
import scipy
from pylab import *
import matplotlib
matplotlib.use("TkAgg")
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

def extract(img):
    if type(img) is not np.ndarray:
        image = cv2.imread(img)
    else: image = img
    #image = Image.open(img)
    #image
    im = image#.convert('L')
    # create a new figure
    figure()
    gray()
    # show contours with origin upper left corner
    #contour(im, origin='image')
    axis('equal')
    axis('off')
    im_array = array(im)
    figure()

    hist(im_array.flatten(), 128)
    show()
    figure()
    #p = image.convert("L").filter(ImageFilter.GaussianBlur(radius=2))
    #p.show()
    return hist

def extraction(img):
    if type(img) is not np.ndarray:
        image = cv2.imread(img)
    else:
        image = img
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    #image = cv2.imread(img)
    chans = cv2.split(image) #coloured
    colors = ("b", "g", "r")#coloured
    #plt.figure()
    #plt.title("'Flattened' Color Histogram")
    #plt.xlabel("Bins")
    #plt.ylabel("# of Pixels")
    features = []
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #istogramma = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32],[0, 256, 0, 256, 0, 256])
    for (chan, color) in zip(chans, colors): #coloured
        #create a histogram for the current channel and
        #concatenate the resulting histograms for each
        #channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])#coloured
        features.extend(hist)#coloured
    #plt.plot(istogramma)
    #plt.show()
    #plt.plot(hist, color = color)
    #plt.xlim([0, 256])
    #plt.show()
    istogramma = np.array(features).flatten()#coloured#cv2.normalize(istogramma, istogramma).flatten() #
    #plot(istogramma)

    #istogramma.show()
    return istogramma


def extractionshow(img):
    if type(img) is not np.ndarray:
        image = cv2.imread(img)
    else:
        image = img
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    #image = cv2.imread(img)
    chans = cv2.split(image) #coloured
    colors = ("b", "g", "r")#coloured
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    features = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #istogramma = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32],[0, 256, 0, 256, 0, 256])
    for (chan, color) in zip(chans, colors): #coloured
        #create a histogram for the current channel and
        #concatenate the resulting histograms for each
        #channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])#coloured
        features.extend(hist)#coloured
        #plt.plot(istogramma)
        #plt.show()
        plt.plot(hist, color = color)
        plt.xlim([0, 256])
    plt.show()
    istogramma = np.array(features).flatten()#coloured#cv2.normalize(istogramma, istogramma).flatten() #
    #cv2.waitKey(0)
    #plot(istogramma)

    #istogramma.show()
    return plt


def hogextraction(img):
    if type(img) is not np.ndarray:
        image = cv2.imread(img)
    else:
        image = img
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True, multichannel=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    return fd,hog_image


def match(img1,img2):
    #	image1 = cv2.imread("moto1.jpg")
    #	image2 = cv2.imread("moto1.jpg")
    if type(img1) is not np.ndarray:
        img1 = cv2.imread(img1)
    if type(img2) is not np.ndarray:
        img2 = cv2.imread(img2)

    if(img1.shape>img2.shape):
        height, width, channels = img1.shape
        img2 = cv2.resize(img2,(width,height))
    if (img1.shape<img2.shape):
        height, width, channels = img2.shape
        img1 = cv2.resize(img1,(width,height))
    colhist1 = extraction(img1)
    colhist2 = extraction(img2)
    result = cv2.compareHist(colhist1, colhist2, cv2.HISTCMP_BHATTACHARYYA)
    sim = similarity(img1,img2)
    #dist_l2 = hammingDistance(hoghist1,hoghist2)#np.linalg.norm(hoghist1-hoghist2)
    #print(dist_l2)
    return result,sim




def hammingDistance(img1,img2):
    if type(img1) is not np.ndarray:
        img1 = cv2.imread(img1)
    if type(img2) is not np.ndarray:
        img2 = cv2.imread(img2)
    if (img1.shape > img2.shape):
        height, width, channels = img1.shape
        img2 = cv2.resize(img2, (width, height))
    if (img1.shape < img2.shape):
        height, width, channels = img2.shape
        img1 = cv2.resize(img1, (width, height))
    score =  scipy.spatial.distance.hamming(img1.flatten(),img2.flatten())
    print(score)
    return  score

def similarity(img1,img2):
    if type(img1) is not np.ndarray:
        img1 = cv2.imread(img1)
    if type(img2) is not np.ndarray:
        img2 = cv2.imread(img2)
    matches = []
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(img1, None)
    kp_2, desc_2 = sift.detectAndCompute(img2, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    if (len(kp_1) >= 2 and len(kp_2) >= 2):
        matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    ratio = 0.475
    result = 0
    if len(matches)>0:
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_points.append(m)
                result = (len(good_points))

    #conf = cv2.drawMatches(img1, kp_1, img2, kp_2, good_points, None)
    #cv2.imshow("result", conf)
    #cv2.imshow("Original", img1)
    #cv2.imshow("Duplicate", img2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(result)
    return result

def direction(x1,y1,x2,y2,img_center):
    if(img_center[0] != 0 and img_center[1]!= 0):
        x1 = x1 - img_center[0]
        y1 = img_center[1] - y1
        x2 = x2 - img_center[0]
        y2 = img_center[1] - y2
    x2 = x2 - x1
    y2 = y2 - y1
    angolo = math.degrees(math.atan2(y2, x2))
    if angolo < 0:
        angolo = 360 + angolo
    return angolo

#angle = 180 - abs(abs(350 - 210) - 180);

#directionMatch(700,200,100,200,[640,360])
#print(match("m1.png","m2.png"))
#contours("ret.jpg")
#print(match("white.png","black.png"))
#f=open("times.extract","txt+")
#hogextraction("screen.jpg")
#hogextraction("prova.jpg")
#hammingDistance("prova6.jpg","prova6.jpg")
#image = cv2.imread("screen.jpg")
#plt.hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')  # calculating histogram


