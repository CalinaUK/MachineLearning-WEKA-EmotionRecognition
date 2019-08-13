import random
from linereader import getline
from config import data_files, data_dirs, data_files3, data_dirs3, NUM_LINES_RAND_CSV, NUM_OF_RAND_CSV, NUM_LINES_TRAIN_CW3, NUM_LINES_TEST_CW3, LINE_SPLIT_BY
from scripts.wekaloader import convert_file

# Putput Predictions to file
def output_pred(pout, output_file):
    with output_file.open("w+", encoding ="utf-8") as f:
        f.write(str(pout))


#Output Evaluation Summary to file
def output_eval(result, output_file):
    with output_file.open("w+", encoding ="utf-8") as f:
        f.write(str(result.summary()))
        f.write("\n")
        f.write(str(result.class_details()))
        f.write("\n")
        f.write(str(result.confusion_matrix))

def output_cluster(result, output_file):
    with output_file.open("w+", encoding ="utf-8") as f:
        f.write(str(result.cluster_results))


# Output Selected Attributes to file
def output_select_attribute(attsel, output_file):
    with output_file.open("w+", encoding ="utf-8") as f:
        f.write("# attributes: " + str(attsel.number_attributes_selected))
        f.write("\n")
        f.write("attributes: " + str(attsel.selected_attributes))
        f.write("\n")
        f.write("result string:\n" + attsel.results_string)


# Returns all .arff files in a directory
def get_ARFFs_in_dir(dir):
    dir = dir.glob("*")
    arffs = [x for x in dir if (x.is_file() and x.parts[-1].endswith(".arff"))]
    return arffs

def get_CSVs_in_dir(dir):
    dir = dir.glob("*")
    csvs = [x for x in dir if (x.is_file() and x.parts[-1].endswith(".csv"))]
    return csvs

def splitTestingSets(name):
    # Check file is csv
    if not name.endswith(".csv"):
        print("%s not CSV file." % name)
        return

    main_file = name[:-4]                   # Removes '.csv'
    data_directory = main_file + "_data"    # Data directory working in
    filepath = data_dirs3[data_directory]   # Gets filepath
    name = data_files3[name]                # Get name of file
    files_created = []                      # Array of files converted
    seenImgs = []                           # Rows in  Training CSV

    linecount = NUM_LINES_TEST_CW3   # Set linecount size
    num_of_files = 1
    linecounter = [linecount-2]
    filename = ["_rows_testing_set"]


    # Split if training set
    if(main_file.startswith("fer2017-training")):
        # Set linecount size
        linecount = NUM_LINES_TRAIN_CW3

        # Split into 2 files
        num_of_files = 2

        # Difference between total line linecounter
        # and the amount of lines to split by
        linecount_diff = linecount - LINE_SPLIT_BY - 2

        # array to differenciate
        linecounter = [LINE_SPLIT_BY, linecount_diff]
        filename=["_" + str(LINE_SPLIT_BY) + "_rows_training_set", "_" + str(linecount_diff) + "_rows_training_set"]


    for i in range(2, linecount):   # Need to skip first line
        seenImgs.append(i)

    random.shuffle(seenImgs)

    mapHappy ={
        "Happy" : "1",
        "NotHappy" : "0"
    }

    # Overwrite all files?
    askUser = False
    canWriteAll = False


    # Split CSV into two files
    for j in range(num_of_files):
        # you need to change to the name of the file you want to create
        stringFile = main_file + filename[j]+ ".csv"
        my_file = filepath / stringFile

        if my_file.is_file():
            while True:
                if not canWriteAll and not askUser:
                    userInput = input(
                        "Do you wish to overwrite all %s random files? y/n \n" % main_file)
                    if userInput == 'y':
                        canWriteAll = True
                        askUser = True
                        f = my_file.open(mode='w')
                        break
                    elif userInput == 'n':
                        askUser = True
                        break
                    else:
                        print("Please enter only  y/n/all")
                else:
                    f = my_file.open(mode='w')
                    break
        else:
            canWriteAll = True
            f = my_file.open(mode="x")

        if(canWriteAll):
            print("Creating file %s..." % stringFile)

            f.write("emotion")
            for x in range(1, 2305):
                f.write(",pixel" + str(x))
            f.write("\n")

            # NUM_LINES_RAND_CSV - Change in config.py
            for i in range(linecounter[j]):
                line_to_add = getline(name, seenImgs.pop())
                firstValue = line_to_add.split(",")

                if(main_file.endswith("happy")):
                    size = len(firstValue)-1    # Get last element in line
                    f.write(mapHappy[firstValue[size].rstrip("\n")] + ",")

                    splitPixels = firstValue[:-1]
                    for x in range(0, len(splitPixels)):
                        f.write(splitPixels[x])
                        if x < len(splitPixels)-1:
                            f.write(",")
                        else:
                            f.write("\n")
                else:
                    f.write(firstValue[0] + ",")
                    splitPixels = firstValue[1].split(" ")

                    for x in range(0, len(splitPixels)):
                        f.write(splitPixels[x])
                        if x < len(splitPixels)-1:
                            f.write(",")

            files_created.append(my_file)
            print("Created")
    return files_created

# Randomise and preprocess CSV's
def randomise_CSV(name):
    linecount = NUM_LINES_TEST_CW3   # Set linecount size
    num_of_files = 1

    # Split if training set
    if(name.startswith("fer2017-training")):
        # Set linecount size
        linecount = NUM_LINES_TRAIN_CW3

    seenImgs = []       # Rows in CSV
    for i in range(2, linecount + 1):   # Need to skip first line
        seenImgs.append(i)

    random.shuffle(seenImgs)

    mapHappy ={
        "Happy" : "1",
        "NotHappy" : "0"
    }


    # Overwrite all files?
    askUser = False
    canWriteAll = False

    # Check file is csv
    if not name.endswith(".csv"):
        print("%s not CSV file." % name)
        return

    main_file = name[:-4]                   # Removes '.csv'
    data_directory = main_file + "_data"    # Data directory working in
    filepath = data_dirs3[data_directory]    # Gets filepath
    name = data_files3[name]                 # Get name of file
    files_created = []                      # Array of files converted

    # create 10 files
    for x in range(1, num_of_files + 1):
        # you need to change to the name of the file you want to create
        stringFile = name.name
        my_file = filepath / stringFile

        if my_file.is_file():
            while True:
                if not canWriteAll and not askUser:
                    userInput = input(
                        "Do you wish to overwrite all %s random files? y/n \n" % main_file)
                    if userInput == 'y':
                        canWriteAll = True
                        askUser = True
                        f = my_file.open(mode='w')
                        break
                    elif userInput == 'n':
                        askUser = True
                        break
                    else:
                        print("Please enter only  y/n/all")
                else:
                    f = my_file.open(mode='w')
                    break
        else:
            canWriteAll = True
            f = my_file.open(mode="x")

        if(canWriteAll):
            print("Creating file %s..." % stringFile)

            f.write("emotion")
            for x in range(1, 2305):
                f.write(",pixel" + str(x))

            f.write("\n")

            # NUM_LINES_RAND_CSV - Change in config.py
            for i in range(linecount-2):
                line_to_add = getline(name, seenImgs.pop())
                firstValue = line_to_add.split(",")

                if(main_file.endswith("happy")):
                    size = len(firstValue)-1    # Get last element in line
                    f.write(mapHappy[firstValue[size].rstrip("\n")] + ",")

                    splitPixels = firstValue[:-1]
                    for x in range(0, len(splitPixels)):
                        f.write(splitPixels[x])
                        if x < len(splitPixels)-1:
                            f.write(",")
                        else:
                            f.write("\n")
                else:
                    f.write(firstValue[0] + ",")
                    splitPixels = firstValue[1].split(" ")

                    for x in range(0, len(splitPixels)):
                        f.write(splitPixels[x])
                        if x < len(splitPixels)-1:
                            f.write(",")

            files_created.append(my_file)
            print("Created")
    return files_created


def to_ARFF(file):
    if file.is_file():
        # Edits Path from CSV to ARFF file type
        arff = file.parents[0]/((file.parts[-1][:-3])+"arff")

        fp = file.absolute()
        out_fp = arff.absolute()

        convert_file(fp, out_fp)
        return arff
