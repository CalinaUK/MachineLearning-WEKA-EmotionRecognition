from config import jvm_config
import weka.core.jvm as jvm
import weka.core.converters as converters
import weka.core.dataset as dataset


def load_Arff_file(file):
    if type(file) is not str:
        file = str(file)

    loader = converters.Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(file)
    return data



def convert_file(from_x, to_y):
    # Create nominals for emotion attr
    value_list = []
    for i in range(2): value_list.append(str(i))
    # Check if more nominals needed
    if not from_x.parent.name.endswith("happy_data"):
        for i in range(2, 7): value_list.append(str(i))

    if type(from_x) is not str:
        from_x = str(from_x)
    if type(to_y) is not str:
        to_y = str(to_y)

    # Loads data based on file type
    data = converters.load_any_file(from_x)
    # emotion attribute located at index 0
    emotion_atr = data.attribute(0)


    # need emotion attr to be nominal
    if not emotion_atr.is_nominal:
        # Modify emotion attr
        emotion_atr = emotion_atr.create_nominal(emotion_atr.name, value_list)

        # Store all emotion values before swapping
        # to modified emotion_atr
        emotion_vals = []
        for i in dataset.InstanceIterator(data):
            emotion_vals.append(int(i.get_value(0)))

        # Replace emotion attr
        data.delete_first_attribute()
        data.insert_attribute(emotion_atr, 0)

        # Set the values in new emotion attr
        for i in dataset.InstanceIterator(data):
            i.set_string_value(0, str(emotion_vals.pop(0)))


    converters.save_any_file(data, to_y)

def stop_weka():
    jvm.stop()

def start_weka():
    jvm.start(**jvm_config)
