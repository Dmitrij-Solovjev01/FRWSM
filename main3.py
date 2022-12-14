import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint

AImg = [[cv2.imread('./archive/s' + str(person) + '/' + str(img) + '.pgm', cv2.IMREAD_GRAYSCALE)
         for img in range(1, 11)] for person in range(1, 41)]

ParHist = 256
ParRand = 300
ratio = 0.5

AInd = np.array([])
for i in range(0, 40):
    if len(AInd) == 0:
        AInd = [[AImg[i][1], AImg[i][7]]]           # Scale 8
    else:
        AInd = np.append(AInd, [[AImg[i][1], AImg[i][7]]], axis=0)


mask = np.zeros(AImg[0][0].shape[:2], dtype="uint8")
for i in range(ParRand):
    mask[randint(0, 111)][randint(0, 91)] = 255


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
        masked = cv2.bitwise_and(img, img, mask=mask)

        a_help = []
        for k in range(0, 112):
            for m in range(0, 92):
                if mask[k][m]:
                    a_help.append(masked[k][m])

        return a_help

data = [[0, 0 ,0]]

for num_of_person in range(1, 5):
    data_help = np.array([])
    for method in range(3):
        tr = 0
        for Bperson in range(0, num_of_person):
            for img in range(0, 10):
                Amin = [[0] * 10 for i in range(num_of_person)]
                for person in range(0, num_of_person):
                    err = []
                    for reference in range(len(AInd[0])):
                        err = np.append(err, compare(
                            select_method(method, AInd[person][reference]),
                            select_method(method, AImg[Bperson][img]), 0))
                    Amin[person] = [np.min(err), np.argmin(err)]

                result = 0
                for i in range(len(Amin)):
                    if Amin[i][0] < Amin[result][0]:
                        result = i
                # print(str(Bperson + 1) + " recogn. as " + str(result + 1) + " with index: " + str(Amin[result][1]))

                if result == Bperson:
                    tr += 1
                else:
                    if False:
                        fig = plt.figure(figsize=(12, 8))
                        columns = 3
                        rows = 2

                        # ax enables access to manipulate each of subplots
                        ax = []
                        arr = [AImg[Bperson][img], AInd[Bperson][0], AInd[result][Amin[result][1]]]

                        arr_name = ["TRUE ETALON", "TEST IMAGE", "FALSE ETALON"]
                        arr_type_name = ["Scale", "Hist", "Random"]
                        arr_plot = [cv2.calcHist([AImg[Bperson][img]], [0], None, [ParHist], [0, ParHist]),
                                    cv2.calcHist([AInd[Bperson][0]], [0], None, [ParHist], [0, ParHist]),
                                    cv2.calcHist([AInd[result][Amin[result][1]]], [0], None, [ParHist], [0, ParHist])]

                        for i in range(columns):
                            ax.append(fig.add_subplot(rows, columns, i + 1))
                            ax[-1].set_title(arr_name[i])  # set title
                            plt.imshow(arr[i], alpha=1, cmap='gray')

                        for i in range(columns):
                            ax.append(fig.add_subplot(rows, columns, i + 4))
                            plt.xlabel("Bins")
                            plt.ylabel("# of Pixels")
                            plt.plot(arr_plot[i], alpha=1)
                            plt.xlim([0, 256])
                            plt.ylim([0, 150])

                        plt.suptitle(arr_type_name[method], fontsize=16)
                        plt.show()

        #print((tr - num_of_person * len(AInd[0])) / (num_of_person * 10 - num_of_person * len(AInd[0])) * 100)
        data_help = np.append(data_help, (tr - num_of_person * len(AInd[0])) / (num_of_person * 10 - num_of_person * len(AInd[0])) * 100)
    data = np.append(data, [data_help], axis=0)
    #print("ITERATION: "+str(num_of_person)+"------------------------------------------")

#print(data)
#print(data.transpose())

#help_array =[]
#fig = plt.figure(figsize=(12, 8))

#help_array.append(fig.add_subplot(1, 1, 1))
plt.plot(data.transpose()[0][1::], alpha=0.5)

#help_array.append(fig.add_subplot(1, 1, 1))
plt.plot(data.transpose()[1][1::], alpha=0.6)

#help_array.append(fig.add_subplot(1, 1, 1))
plt.plot(data.transpose()[2][1::], alpha=1)
plt.xlabel("persons")
plt.ylabel("accuracy")
plt.legend(["scale", "hist", "rand point"])

plt.show()

'''
plt.xlabel("persons")
plt.ylabel("accuracy")
plt.plot(data.transpose()[0][1::])
plt.show()
plt.xlabel("persons")
plt.ylabel("accuracy")
plt.plot(data.transpose()[1][1::])
plt.show()
plt.xlabel("persons")
plt.ylabel("accuracy")
plt.plot(data.transpose()[2][1::])
plt.show()
'''
