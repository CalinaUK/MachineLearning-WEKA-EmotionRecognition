from pathlib import Path


# Lines per random CSV File
NUM_LINES_RAND_CSV = 1000
NUM_OF_RAND_CSV = 1

# CW3 File configs
NUM_LINES_TRAIN_CW3 = 28700
NUM_LINES_TEST_CW3 = 7100
LINE_SPLIT_BY = 9000


# configs for JVM
jvm_config = {
    "system_cp": True,
    "packages": True,
    "max_heap_size": "4000m"
}


# configs for Weka
data_dir3 = Path("./CW3-2017")
data_dirs3 = {
    "root": data_dir3,
    "fer2017-training_data":    data_dir3 / "fer2017-training_data/",
    "fer2017-training-happy_data":  data_dir3 / "fer2017-training-happy_data/",
    "fer2017-testing_data":  data_dir3 / "fer2017-testing_data/",
    "fer2017-testing-happy_data":  data_dir3 / "fer2017-testing-happy_data/",
    "fer2017-default-NOsplit":    data_dir3 / "fer2017-default-NOsplit/",
    "fer2017-split-add-9000":  data_dir3 / "fer2017-split-add-9000/",
    "fer2017-split-add-16000":  data_dir3 / "fer2017-split-add-16000/",
}
data_files3 = {
    "fer2017-training.csv":         data_dir3 / "fer2017-training.csv",
    "fer2017-testing.csv":          data_dir3 / "fer2017-testing.csv",
    "fer2017-training-happy.csv":   data_dir3 / "fer2017-training-happy.csv",
    "fer2017-testing-happy.csv":    data_dir3 / "fer2017-testing-happy.csv"
}
data_dirs3_appended = {
    "root": data_dir3,
#    "fer2017-default-NOsplit-all":  data_dirs3["fer2017-default-NOsplit"] / "all_emotions/",
#    "fer2017-default-NOsplit-happy":  data_dirs3["fer2017-default-NOsplit"] / "happy_emotions/",
#    "fer2017-split-add-9000-all":  data_dirs3["fer2017-split-add-9000"] / "all_emotions/",
#    "fer2017-split-add-9000-happy":  data_dirs3["fer2017-split-add-9000"] / "happy_emotions/",
#    "fer2017-split-add-16000-all":  data_dirs3["fer2017-split-add-16000"] / "all_emotions/",
   "fer2017-split-add-16000-happy":  data_dirs3["fer2017-split-add-16000"] / "happy_emotions/",
}


# configs for Weka
data_dir = Path("./fer2018")
data_dirs = {
    "root": data_dir,
    "fer2018_data": data_dir / "fer2018_data/",
    "fer2018angry_data": data_dir / "fer2018angry_data/",
    "fer2018disgust_data": data_dir / "fer2018disgust_data/",
    "fer2018fear_data": data_dir / "fer2018fear_data/",
    "fer2018happy_data": data_dir / "fer2018happy_data/",
    "fer2018neutral_data": data_dir / "fer2018neutral_data/",
    "fer2018sad_data": data_dir / "fer2018sad_data/",
    "fer2018surprise_data": data_dir / "fer2018surpise_data/"
}
data_files = {
    "fer2018.csv": data_dir / "fer2018.csv",
    "fer2018angry.csv": data_dir / "fer2018angry.csv",
    "fer2018disgust.csv": data_dir / "fer2018disgust.csv",
    "fer2018fear.csv": data_dir / "fer2018fear.csv",
    "fer2018happy.csv": data_dir / "fer2018happy.csv",
    "fer2018neutral.csv": data_dir / "fer2018neutral.csv",
    "fer2018sad.csv": data_dir / "fer2018sad.csv",
    "fer2018surprise.csv": data_dir / "fer2018surprise.csv"
}
