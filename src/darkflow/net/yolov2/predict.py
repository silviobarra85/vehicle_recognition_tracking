import numpy as np
import math
import cv2
import os
import json
from PIL import Image
from PIL import ImageFilter
from pylab import *
from PIL import *
from . import featureext
from ..help import commonList
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor

def expit(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
    # meta
    meta = self.meta
    boxes = list()
    boxes=box_constructor(meta,net_out)
    return boxes

def postprocess(self,nobg, net_out, im, save = True):
    """
    Takes net output, draw net_out, save to disk
    """
    boxes = self.findboxes(net_out)

    # meta
    meta = self.meta
    threshold = meta['thresh']
    colors = meta['colors']
    labels = meta['labels']
    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else: imgcv = im.copy()
    h, w, _ = imgcv.shape
    recognized = []
    founded = []
    boxindex = 0
    resultsForJSON = []
    for el in commonList.recognized:
        recognized.append(el)
    #scorre i box trovati nel frame (quindi tutti gli oggetti identificati)
    #cv2.imshow("prima", im)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    imctr = im.shape
    image_center = [None,None]
    image_center[0] = imctr[0]/2
    image_center[1] = imctr[1]/2
    for b in boxes:

        #print("immagine numero: " +str(boxindex))
        boxResults = self.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        #boxindex = boxindex + 1
        left, right, top, bot, mess , max_indx, confidence = boxResults
        crop_img = im[top:bot,left:right] #AA
        center = [bot-top,right-left]

        indice = 0
        #vector = array(crop_img)
        #cropped = cv2.imwrite('prova.jpg', crop_img) #AA
        #se distanza tra istogrammi classici: 0.5 - 0.75
        #cv2.imshow(str(cropped), crop_img)  # AA
        #cv2.waitKey(0)  # AA
        if len(recognized) != 0:
            found = False
            accettabili = []
            index = 0
            for saved in recognized:
                result = featureext.match (crop_img,saved[0])
                if(result[0] <= 0.35): #valore ottimo, almeno per ora -> 0.35
                    found = True

                    accettabili.append([index, result[1],saved[1]-1,crop_img, ])
                index = index +1
            #str(commonList.recognized.index(crop_img))
            if (found == False):
                commonList.trovati = commonList.trovati + 1
                # cv2.imwrite('prova'+str(commonList.trovati)+'.jpg', crop_img) #AA

                commonList.recognized.append(
                       [crop_img, commonList.trovati, mess, commonList.tempo, "fine del video",center,None])
                recognized.append([crop_img, commonList.trovati, mess, commonList.tempo, "fine del video",center,None])
                mess = mess + str(commonList.trovati)
                print('Aggiunto: ' +mess+ " -> colore non corrispondente ")
            if(found == True):
                max = 0
                bestRecognized = 0
                mainListfounded = 99999
                for element in accettabili:
                    if element[1]>max:
                        found = True
                        max = element[1]
                        bestRecognized = element[0]
                        mainListfounded = element[2]
                        newimage = element[3]
                        dist = math.sqrt((center[0] - commonList.recognized[mainListfounded][5][0]) ** 2 + (
                                    center[1] - commonList.recognized[mainListfounded][5][1]) ** 2)
                        if commonList.recognized[mainListfounded][6] is not None:
                            direction_diff = 180- abs(abs(commonList.recognized[mainListfounded][6]- featureext.direction(commonList.recognized[mainListfounded][5][0],commonList.recognized[mainListfounded][5][1],center[0],center[1],image_center)) - 180)

                if (mainListfounded != 99999 or max!=0) and (commonList.recognized[mainListfounded][6] is None or direction_diff <= 210):

                    print(mess+str(commonList.recognized[mainListfounded][1]) + " : " + str(max) + " dist: "  + str(dist))
                    #cv2.imshow(mess+str(commonList.recognized[mainListfounded][1]), newimage)
                    #cv2.imshow("salvata", commonList.recognized[mainListfounded][0])
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    commonList.recognized[mainListfounded][0] = newimage
                    commonList.recognized[mainListfounded][4] = commonList.tempo
                    commonList.recognized[mainListfounded][5] = center
                    commonList.recognized[mainListfounded][6] = featureext.direction(commonList.recognized[mainListfounded][5][0],commonList.recognized[mainListfounded][5][1],center[0],center[1],image_center)
                    mess = mess+str(commonList.recognized[mainListfounded][1])
                    del recognized[bestRecognized]
                else:
                    commonList.trovati = commonList.trovati + 1
                    # cv2.imwrite('prova'+str(commonList.trovati)+'.jpg', crop_img)
                    mess = mess + str(commonList.trovati)
                    commonList.recognized.append(
                        [crop_img, commonList.trovati, mess, commonList.tempo, "fine del video",center,None])
                    recognized.append([crop_img, commonList.trovati, mess, commonList.tempo, "fine del video",center,None])
                    print('Aggiunto: ' +mess+ " -> colore ok, ma nessun punto in comune")



        else:
            commonList.trovati = commonList.trovati + 1
            #cv2.imwrite('prova'+str(commonList.trovati)+'.jpg', crop_img)
            mess = mess + str(commonList.trovati)
            commonList.recognized.append([crop_img,commonList.trovati,mess,commonList.tempo,"fine del video",center,None])
            recognized.append([crop_img,commonList.trovati,mess,commonList.tempo,"fine del video",center,None])
            print('aggiunto: lista vuota')

        thick = int((h + w) // 300)
        founded.append([mess, left, top, right, bot, max_indx, thick])

    #cv2.imshow("dopo", im)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
    for b in boxes:
        # print("immagine numero: " +str(boxindex))
        boxResults = self.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        # boxindex = boxindex + 1
        left, right, top, bot, mess, max_indx, confidence = boxResults
        if self.FLAGS.json:
            resultsForJSON.append({"label": founded[boxindex][0], "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
            continue

        cv2.rectangle(imgcv,
                      (left, top), (right, bot),
                      colors[max_indx], founded[boxindex][6])
        cv2.putText(imgcv,founded[boxindex][0], (left, top - 12),
                    0, 1e-3 * h, colors[max_indx],founded[boxindex][6]//3)
        boxindex = boxindex +1

    if not save: return imgcv

    outfolder = os.path.join(self.FLAGS.imgdir, 'out')
    img_name = os.path.join(outfolder, os.path.basename(im))
    if self.FLAGS.json:
        textJSON = json.dumps(resultsForJSON)
        textFile = os.path.splitext(img_name)[0] + ".json"
        with open(textFile, 'w') as f:
            f.write(textJSON)
        return

    cv2.imwrite(img_name, imgcv)
