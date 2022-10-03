import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint
import dlib
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
ParHist = 256
ParRand = 400


AImg = [
    [cv2.imread('./archive/s' + str(person) + '/' + str(img) + '.pgm', cv2.IMREAD_GRAYSCALE) for img in range(1, 11)]
    for person in range(1, 41)]

faces = face_detector(AImg[0][0], 1)

landmark_tuple = []

for k, d in enumerate(faces):
   landmarks = landmark_detector(AImg[0][0], d)
   for n in range(0, 27):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmark_tuple.append((x, y))
#      cv2.circle(AImg[0][0], (x, y), 2, (255, 255, 0), -1)

routes = []
def Ulala(img):
    landmark_tuple = []
    for k, d in enumerate(faces):
        landmarks = landmark_detector(img, d)
        for n in range(0, 27):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_tuple.append((x, y))
            #cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

    routes = []

    for i in range(10, 3, -1):
        from_coordinate = landmark_tuple[i + 1]
        to_coordinate = landmark_tuple[i]
        routes.append(from_coordinate)

    from_coordinate = landmark_tuple[4]
    to_coordinate = landmark_tuple[14]
    routes.append(from_coordinate)

    for i in range(17, 20):
        from_coordinate = landmark_tuple[i]
        to_coordinate = landmark_tuple[i + 1]
        routes.append(from_coordinate)

    from_coordinate = landmark_tuple[19]
    to_coordinate = landmark_tuple[24]
    routes.append(from_coordinate)

    for i in range(24, 26):
        from_coordinate = landmark_tuple[i]
        to_coordinate = landmark_tuple[i + 1]
        routes.append(from_coordinate)

    from_coordinate = landmark_tuple[26]
    to_coordinate = landmark_tuple[12]
    routes.append(from_coordinate)
    routes.append(to_coordinate)

    mask = np.zeros(img.shape[:2], dtype="uint8")
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    return cv2.bitwise_and(img, img, mask=mask)

AImgMod = [[Ulala(img) for img in person] for person in AImg]

#for i in range(0, 40):
#    for j in range(0, 10):
#        plt.imshow(AImgMod[i][j])
#        plt.show()

#AHist = [[cv2.calcHist([img], [0], None, [ParHist], [0, ParHist]) for img in person] for person in AImgMod]

cv2.imshow("asd2", Ulala(AImgMod[0][0]))





mask = np.zeros(AImg[0][0].shape[:2], dtype="uint8")
for i in range(ParRand):
    mask[randint(0, 111)][randint(0, 91)] = 255

ARand = []

for i in range(0, 40):
    AHelp2 = []
    for j in range(0, 10):
        masked = cv2.bitwise_and(AImg[i][j], AImg[i][j], mask=mask)
        AHelp = []
        for k in range(0, 112):
            for l in range(0, 92):
                if mask[k][l]:
                    AHelp.append(masked[k][l])
        AHelp2.append(AHelp)
    ARand.append(AHelp2)


def Scale(img):
    small_to_large_image_size_ratio = 0.5
    return np.array(cv2.resize(img,  # original image
                               (0, 0),  # set fx and fy, not the final size
                               fx=small_to_large_image_size_ratio,
                               fy=small_to_large_image_size_ratio)).flatten()


''',interpolation=cv2.INTER_NEAREST'''

AScale = [[Scale(img) for img in person] for person in AImg]


# print(len(AScale[0][0]))

# print(AScale)
# plt.imshow(np.resize(np.array(AScale[0][0]),(11,9)),  cmap='gray')
# plt.show()


def compare(h1, h2):
    sum = 0
    # print(h1)

    if len(h1) != len(h2):
        print("ALARM!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        for j in range(len(h1)):
            sum += abs(int(h1[j]) - int(h2[j]))  # **2
    return sum


def compare2(h1, h2):
    sum = 0
    if len(h1) != len(h2):
        print("ALARM!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        for j in range(len(h1)):
            sum += (h1[j] - h2[j]) ** 2
    return sum


def compare3(h1, h2):
    sum = 0
    if len(h1) != len(h2):
        print("ALARM!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        for j in range(len(h1)):
            sum += abs(h1[j] ** 2 - h2[j] ** 2)
    return sum

def compare4(h1, h2):
    sum = 0
    if len(h1) != len(h2):
        print("ALARM!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        for j in range(len(h1)):
            sum += (h1[j] - h2[j]) ** 8
    return sum


AScaleInd =np.array([])
AHistInd = np.array([])

for i in range(0, 40):
#    AHistInd = np.append(AHistInd, 0)
#    min = []
#    for j in range(0, 10):
#        sum = 0
#        for k in range(0, 10):
#            sum += compare(AHist[i][j], AHist[i][k])
#        min = np.append(min, sum)
#        if j == 9:
#            result = 0
#
#            for l in range(len(min)):
#                if min[l] < min[result]:
#                    result = l
#
    if len(AHistInd) == 0:
        AScaleInd = [AScale[i][8]]
            #    AHistInd = [[AImg[i][result], AImg[i][9-result]]]
        AHistInd = [[AImgMod[i][0], AImgMod[i][1]]]

    else:
        AScaleInd = np.append(AScaleInd, [AScale[i][8]], axis=0)
        AHistInd = np.append(AHistInd, [[AImgMod[i][0], AImgMod[i][1]]], axis=0)


tr = 0

for Bperson in range(0, 40):
    for img in range(0, 10):
        Amin = np.array([])
        for person in range(0, 40):
            err = []
            for l in range(len(AHistInd[0])):
                err = np.append(err, compare(cv2.calcHist([AHistInd[person][l]], [0], None, [ParHist], [0, ParHist])[1::],
                                             cv2.calcHist([AImgMod[Bperson][img]], [0], None, [ParHist], [0, ParHist])[1::]))
            Amin = np.append(Amin, np.min(err))

        result = 0
        for i in range(len(Amin)):
            if Amin[i] < Amin[result]:
                result = i
        print(str(Bperson + 1) + " recogn. as " + str(result + 1))


        if result == Bperson:
            tr += 1
        else:
            err = []
            for l in range(len(AHistInd[0])):
                err = np.append(err, compare(cv2.calcHist([AHistInd[result][l]], [0], None, [ParHist], [0, ParHist])[1::],
                                             cv2.calcHist([AImgMod[Bperson][img]], [0], None, [ParHist], [0, ParHist])[1::]))
            index = np.where(err == np.min(err))[0][0]
            print(index)

            plt.figure()
            plt.title("TEST IMAGE")
            plt.imshow(AImgMod[Bperson][img], cmap='gray')
            plt.figure()
            plt.title("FALSE ETALON")
            plt.imshow(AHistInd[result][index], cmap='gray')
            plt.figure()
            plt.title("TRUE ETALON")
            plt.imshow(AHistInd[Bperson][0], cmap='gray')

            plt.figure()
            plt.title("FALSE ETALON")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            plt.plot(cv2.calcHist([AHistInd[result][index]], [0], None, [ParHist], [0, ParHist])[1::])
            plt.xlim([0, 256])
            plt.ylim([0, 150])

            plt.figure()
            plt.title("TRUE ETALON")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            plt.plot(cv2.calcHist([AHistInd[Bperson][0]], [0], None, [ParHist], [0, ParHist])[1::])
            plt.xlim([0, 256])
            plt.ylim([0, 150])


            plt.figure()
            plt.title("TEST IMAGE")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            plt.plot(cv2.calcHist([AImgMod[Bperson][img]], [0], None, [ParHist], [0, ParHist])[1::])
            plt.xlim([0, 256])
            plt.ylim([0, 150])
            plt.show()

print((tr - 40*len(AHistInd[0])) / 360 * 100)