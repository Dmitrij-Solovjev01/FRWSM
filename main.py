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
                Ahelper.append(Helper_array_of_result[ind][0])
                Ahelper.append(Helper_array_of_result[ind][1])

            if Counter(Ahelper).most_common()[0][0] == Bperson:
                tr[3] += 1

            if True and not(Helper_array_of_result[0][0] == Bperson and Helper_array_of_result[1][0] == Bperson and
                   Helper_array_of_result[2][0] == Bperson):

                if Counter(Ahelper).most_common()[0][1] == Counter(Ahelper).most_common()[1][1]:
                    print(Counter(Ahelper).most_common())
                    fig = plt.figure(figsize=(12, 8))
                    columns = 4
                    rows = 2

                    arr = [AImg[Bperson][img], AInd[Helper_array_of_result[0][0]][Helper_array_of_result[0][2]],
                           AInd[Helper_array_of_result[1][0]][Helper_array_of_result[1][2]], AInd[Helper_array_of_result[2][0]][Helper_array_of_result[2][2]],
                           AInd[Bperson][0], AInd[Helper_array_of_result[0][1]][Helper_array_of_result[0][3]],
                           AInd[Helper_array_of_result[1][1]][Helper_array_of_result[1][3]], AInd[Helper_array_of_result[2][1]][Helper_array_of_result[2][3]]]

                    arr_name = ["Test image",     "Scale1", "Hist1", "Rand1",
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
