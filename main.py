import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint
from collections import Counter


AImg = [[cv2.imread('./archive/s' + str(person) + '/' + str(img) + '.pgm', cv2.IMREAD_GRAYSCALE)
         for img in range(1, 11)] for person in range(1, 41)]

ParHist = 256
ParRand = 300
ratio = 0.5

AInd = np.array([])
for i in range(0, 40):
    if len(AInd) == 0:
        AInd = [[AImg[i][1], AImg[i][7]]] #[[AImg[i][1]]]
    else:
        AInd = np.append(AInd, [[AImg[i][1], AImg[i][7]]], axis=0) #np.append(AInd, [[AImg[i][1]]], axis=0)


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

#data = [[0, 0 ,0]]

#for num_of_person in range(1, 5):
num_of_person = 40
if True:
    #data_help = np.array([])
    tr = [0, 0, 0, 0]
    for Bperson in range(0, num_of_person):
        for img in range(0, 10):
            Amin = [[0] * 10 for i in range(num_of_person)]

            Helper_array_of_result = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

            for method in range(3):
                for person in range(0, num_of_person):
                    err = []
                    for reference in range(len(AInd[0])):
                        err = np.append(err, compare(
                            select_method(method, AInd[person][reference]),
                            select_method(method, AImg[Bperson][img]), 0))
                    Amin[person] = [np.min(err), np.argmin(err)]

                result = 0
                result2 = 0

                if Amin[0][0] > Amin[1][0]:
                    result = 1
                    result2 = 0
                else:
                    result = 0
                    result2 = 1

                for i in range(2, len(Amin)):
                    if Amin[i][0] < Amin[result][0]:
                        result2 = result
                        result = i
                    elif Amin[i][0] < Amin[result2][0]:
                        result2 = i

                Helper_array_of_result[method][0] = result              #person
                Helper_array_of_result[method][1] = result2

                Helper_array_of_result[method][2] = Amin[result][1]     #index of ethoalon
                Helper_array_of_result[method][3] = Amin[result2][1]

                if Helper_array_of_result[method][0] == Bperson:
                    tr[method] += 1

            Ahelper = []

            for ind in range(len(Helper_array_of_result)):
                Ahelper.append(Helper_array_of_result[ind][0])
                #Ahelper.append(Helper_array_of_result[ind][0])
                #Ahelper.append(Helper_array_of_result[ind][1])

            if Counter(Ahelper).most_common()[0][0] == Bperson:
                tr[3] += 1

            if False and not(Helper_array_of_result[0][0] == Bperson and Helper_array_of_result[1][0] == Bperson and
                   Helper_array_of_result[2][0] == Bperson):
                if Counter(Ahelper).most_common()[0][1] == Counter(Ahelper).most_common()[1][1]:
                    fig = plt.figure(figsize=(12, 8))
                    columns = 4
                    rows = 2

                    arr = [AImg[Bperson][img], AInd[Helper_array_of_result[0][0]][Helper_array_of_result[0][2]],
                           AInd[Helper_array_of_result[1][0]][Helper_array_of_result[1][2]], AInd[Helper_array_of_result[2][0]][Helper_array_of_result[2][2]],
                           AInd[Bperson][0], AInd[Helper_array_of_result[0][1]][Helper_array_of_result[0][3]],
                           AInd[Helper_array_of_result[1][1]][Helper_array_of_result[1][3]], AInd[Helper_array_of_result[2][1]][Helper_array_of_result[2][3]]]

                    arr_name = ["Test image", "Scale1", "Hist1", "Rand1",
                                "Correct answer", "Scale2", "Hist2", "Rand2"]

                    ax = []
                    for ind in range(len(arr)):
                        ax.append(fig.add_subplot(rows, columns, ind + 1))
                        ax[ind].set_title(arr_name[ind])  # set title
                        plt.imshow(arr[ind], alpha=1, cmap='gray')


                    '''for i in range(columns):
                        ax.append(fig.add_subplot(rows, columns, i + 4))
                        plt.xlabel("Bins")
                        plt.ylabel("# of Pixels")
                        plt.plot(arr_plot[i], alpha=1)
                        plt.xlim([0, 256])
                        plt.ylim([0, 150])
                    '''

                    #plt.suptitle(arr_type_name[method], fontsize=16)
                    plt.show()

print((tr[0] - num_of_person * len(AInd[0])) / (num_of_person * 10 - num_of_person * len(AInd[0])) * 100)
print((tr[1] - num_of_person * len(AInd[0])) / (num_of_person * 10 - num_of_person * len(AInd[0])) * 100)
print((tr[2] - num_of_person * len(AInd[0])) / (num_of_person * 10 - num_of_person * len(AInd[0])) * 100)
print((tr[3] - num_of_person * len(AInd[0])) / (num_of_person * 10 - num_of_person * len(AInd[0])) * 100)

#        data_help = np.append(data_help, (tr - num_of_person * len(AInd[0])) / (num_of_person * 10 - num_of_person * len(AInd[0])) * 100)
#    data = np.append(data, [data_help], axis=0)
    #print("ITERATION: "+str(num_of_person)+"------------------------------------------")


'''
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