import numpy as np
import pickle
import os


def delete_data_file():
    # Remove file and re-adjust the file numbers
    file_number = input("enter the file number you wish to delete: ")
    Dtypes = ("body", "head")


    dataIndex = pickle.load(open("collectedData/dataIndex.p", "rb"))
    print(dataIndex)
    newIndex = dataIndex - 1

    for atype in Dtypes:
        os.remove("collectedData/{}/training_data_{}_{}.npy".format(atype, atype, str(file_number)))

        for i in range(int(file_number)+1,dataIndex+1):

            old_file_name = "collectedData/{}/training_data_{}_{}.npy".format(atype, atype, str(i))
            temp = np.load(old_file_name)
            os.remove(old_file_name)
            new_file_name = "collectedData/{}/training_data_{}_{}.npy".format(atype, atype, str(i-1))
            np.save(new_file_name, temp)

    print(newIndex)
    pickle.dump(newIndex,open("collectedData/dataIndex.p", "wb"))


delete_data_file()
