print("Importing...")
import handtracking as ht
import os
import tensorflow as tf
print("All is imported")

while True:
    type = input("What nn would you like to use?\n1 - small\n2 - medium size\n3 - big\n:: ")
    try:
        type = int(type)
    except ValueError:
        print("\nPrint int, please\n")

    if type == 1:
        print("\nChoose nn from list")
        files = os.listdir("nets/spr/small")
        for i in range(len(files)):
            print(str(i) + " :: " + files[i])
        print(str(len(files)) + " :: Create new nn")

        i = input(":: ")
        try:
            i = int(i)
        except ValueError:
            print("\nPrint int, please\n")

        if i == len(files):
            nn_name = input("Print nn name\n:: ")
            nn_name = nn_name + ".pth"
            nn_path = "nets/spr/small/" + nn_name
            break
        else:
            try:
                nn_name = files[i]
                nn_path = "nets/spr/small/" + nn_name
                break
            except IndexError:
                print("\nThere is no index " + str(i) + " in list\n")

    elif type == 2:
        print("\nChoose nn from list")
        files = os.listdir("nets/spr/medium")
        for i in range(len(files)):
            print(str(i) + " :: " + files[i])
        print(str(len(files)) + " :: Create new nn")

        i = input(":: ")
        try:
            i = int(i)
        except ValueError:
            print("\nPrint int, please\n")

        if i == len(files):
            nn_name = input("Print nn name\n:: ")
            nn_name = nn_name + ".pth"
            nn_path = "nets/spr/medium/" + nn_name
            break
        else:
            try:
                nn_name = files[i]
                nn_path = "nets/spr/medium/" + nn_name
                break
            except IndexError:
                print("\nThere is no index " + str(i) + " in list\n")

    elif type == 3:
        print("\nChoose nn from list")
        files = os.listdir("nets/spr/big")
        for i in range(len(files)):
            print(str(i) + " :: " + files[i])
        print(str(len(files)) + " :: Create new nn")

        i = input(":: ")
        try:
            i = int(i)
        except ValueError:
            print("\nPrint int, please\n")

        if i == len(files):
            nn_name = input("Print nn name\n:: ")
            nn_name = nn_name + ".pth"
            nn_path = "nets/spr/big/" + nn_name
            break
        else:
            try:
                nn_name = files[i]
                nn_path = "nets/spr/big/" + nn_name
                break
            except IndexError:
                print("\nThere is no index " + str(i) + " in list\n")
typedic = {1: 500, 2: 2000, 3: 5000}

model = tf.keras.models.load_model("nets/hand_tracking/main")
ht.start(model, srp_load_path=nn_path, hidden_size=typedic[type])
