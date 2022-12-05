import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint
from collections import Counter
from matplotlib.widgets import Button

ATitleNames = ["Sourse images(Test image)", "Scale images(Ans. 1)", "Histogram images(Ans. 1)",
            "Random point images(Ans. 1)",
            "Correct answer", "Ans. 2", "Ans. 2", "Ans. 2",
            "Scaled Test image", "Scaled Ans. 1", "Сalculated Hist. Ans. 1", "Random ImageMask Ans. 1",
            "Сalculated Hist. Test image", "Scaled Ans. 2", "Сalculated Hist. Ans. 2", "Random ImageMask Ans. 2"]

AImg = []
AWrongAns = []
AImageRefer = []
ImageMask = []

fig = plt.figure(figsize=(12, 8))
columns = 4
rows = 4

ParHist = 256
ParRand = 300
ratio = 0.5
CompareMethod = 0
AIndexRef = [1, 8]

NumOfPerson = 10
CorrAnswCount = [0, 0, 0, 0]


def prepare():
    global AImageRefer, AImg, ImageMask
    AImg = [[cv2.imread('./archive/s' + str(person) + '/' + str(img) + '.pgm', cv2.IMREAD_GRAYSCALE)
             for img in range(1, 11)] for person in range(1, 41)]
    ImageMask = np.zeros(AImg[0][0].shape[:2], dtype="uint8")
    for i in range(ParRand):
        ImageMask[randint(0, 111)][randint(0, 91)] = 255

    for i in range(0, 40):
        Ahelp = []
        for j in range(len(AIndexRef)):
            Ahelp.append((AImg[i][AIndexRef[j]]))

        if len(AImageRefer) == 0:
            AImageRefer = [Ahelp]
        else:
            AImageRefer = np.append(AImageRefer, [Ahelp], axis=0)

def scale(img):
    return np.array(cv2.resize(img, (0, 0), fx=ratio, fy=ratio)).flatten()


def compare(h1, h2, comp_method):
    diff_sum = 0
    if len(h1) != len(h2):
        print("ALARM!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        for j in range(len(h1)):
            if comp_method == 0:
                diff_sum += abs(int(h1[j]) - int(h2[j]))  # **2
            elif comp_method == 1:
                diff_sum += (h1[j] - h2[j]) ** 2
            else:
                diff_sum += abs(h1[j] ** 2 - h2[j] ** 2)
    return diff_sum


def select_method(meth_num, img):
    if meth_num == 0:
        return scale(img)
    elif meth_num == 1:
        return cv2.calcHist([img], [0], None, [ParHist], [0, ParHist])
    else:
        masked = cv2.bitwise_and(img, img, mask=ImageMask)

        a_help = []
        for k in range(0, 112):
            for m in range(0, 92):
                if ImageMask[k][m]:
                    a_help.append(masked[k][m])

        return a_help


def change_image(img, method):
    if method == 0:
        return np.resize(scale(img), (56, 46))
    elif method == 1:
        return select_method(1, img)
    else:
        return cv2.bitwise_and(img, img, mask=ImageMask)


class Index:
    ind = 0
    hax = []
    NotFirst = 0

    def draw(self, arr):
        self.hax = []

        for img in range(len(arr)):
            if img == 10 or img == 12 or img == 14:
                self.hax.append(fig.add_subplot(rows, columns, img + 1))
                self.hax[img].set_title(ATitleNames[img])
                self.hax[img].plot(np.arange(256), arr[img], color="green")
            else:
                self.hax.append(fig.add_subplot(rows, columns, img + 1))
                self.hax[img].set_title(ATitleNames[img])  # set title
                plt.axis('off')
                plt.imshow(arr[img], alpha=1, cmap='gray')

        fig.canvas.draw_idle()

    def next(self, event):
        for i in range(16):
            self.hax[i]

        self.ind += 1
        self.ind = self.ind % len(AWrongAns)
        self.draw(AWrongAns[self.ind])

    def prev(self, event):

        self.ind -= 1
        self.ind = self.ind % len(AWrongAns)
        self.draw(AWrongAns[self.ind])


def main():
    prepare()

    for Bperson in range(0, NumOfPerson):
        for img in range(0, 10):
            Amin = [[0] * 10 for i in range(NumOfPerson)]

            Helper_array_of_result = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]

            for method in range(3):
                for person in range(0, NumOfPerson):
                    err = []
                    for reference in range(len(AImageRefer[0])):
                        err = np.append(err, compare(
                            select_method(method, AImageRefer[person][reference]),
                            select_method(method, AImg[Bperson][img]), CompareMethod))
                    Amin[person] = [np.min(err), np.argmin(err)]

                AminDistance = np.array(Amin).transpose()[0]

                result = np.argmin(AminDistance)
                result2 = np.argmin(AminDistance[AminDistance != AminDistance[result]])

                Helper_array_of_result[method][0] = (result, Amin[result][1])  # (person, index of ethoalon)
                Helper_array_of_result[method][1] = (result2, Amin[result2][1])  # method probability and index+ value

                if Helper_array_of_result[method][0][0] == Bperson:
                    CorrAnswCount[method] += 1

            Ahelper = []

            for ind in range(len(Helper_array_of_result)):
                Ahelper.append(Helper_array_of_result[ind][0][0])
                Ahelper.append(Helper_array_of_result[ind][0][0])
                Ahelper.append(Helper_array_of_result[ind][1][0])

            if Counter(Ahelper).most_common()[0][0] == Bperson:
                CorrAnswCount[3] += 1

            if True and not (
                    Helper_array_of_result[0][0][0] == Bperson and Helper_array_of_result[1][0][0] == Bperson and
                    Helper_array_of_result[2][0][0] == Bperson):

                if Counter(Ahelper).most_common()[0][1] == Counter(Ahelper).most_common()[1][1]:
                    arr = [AImg[Bperson][img], AImageRefer[Helper_array_of_result[0][0][0]][Helper_array_of_result[0][0][1]],
                           AImageRefer[Helper_array_of_result[1][0][0]][Helper_array_of_result[1][0][1]],
                           AImageRefer[Helper_array_of_result[2][0][0]][Helper_array_of_result[2][0][1]],
                           AImageRefer[Bperson][0], AImageRefer[Helper_array_of_result[0][1][0]][Helper_array_of_result[0][1][1]],
                           AImageRefer[Helper_array_of_result[1][1][0]][Helper_array_of_result[1][1][1]],
                           AImageRefer[Helper_array_of_result[2][1][0]][Helper_array_of_result[2][1][1]]]

                    for my_help_index in range(0, 2):
                        arr.append(change_image(arr[my_help_index * 4], my_help_index))
                        for my_help_index2 in range(0, 3):
                            arr.append(change_image(arr[my_help_index * 4 + my_help_index2 + 1], my_help_index2))

                    AWrongAns.append(arr)

    for i in range(len(CorrAnswCount)):
        print((CorrAnswCount[i] - NumOfPerson * len(AImageRefer[0])) /
              (NumOfPerson * 10 - NumOfPerson * len(AImageRefer[0])) * 100)

    callback = Index()
    axes = plt.axes([0.81, 0.000001, 0.1, 0.075])
    bnext = Button(axes, 'Next', color="gray")
    bnext.on_clicked(callback.next)

    axes = plt.axes([0.7, 0.000001, 0.1, 0.075])
    bprev = Button(axes, 'Prev', color="gray")
    bprev.on_clicked(callback.prev)

    callback.draw(AWrongAns[0])
    plt.show()

if __name__ == "__main__":
    main()
