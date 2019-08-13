from weka.classifiers import Classifier, Evaluation, PredictionOutput
import weka.plot.classifiers as plcls
from weka.core.classes import Random
from scripts.wekaloader import load_Arff_file
from helper import output_eval, output_pred
import time

def run_multilayerPercepton(file, file2=None):
    # Get filename from Pathlib object
    filename = file.parts[-1]
    dir = file.parents[0]

    print("Running Multilayer Percepton on %s" % filename)

    if not filename.endswith(".arff"):
        print("%s not ARFF file." % filename)
        return

    # Removes '.arff' from filename
    filename_base = filename[:-5]

    print("loading data...")
    # Load data with class as first attr
    data = load_Arff_file(file)
    data.class_is_first()

    # If 2nd file load that data too
    if file2:
        print("Loading test...")
        test = load_Arff_file(file2)
        test.class_is_first()

    file_names = [
        "MP_N-500_default_H-1",

        "MP_N-500_H-3",
        "MP_N-500_H-5",
        "MP_N-500_H-7",
        "MP_N-500_H-3-5",
        "MP_N-500_H-5-3",
        "MP_N-500_H-3-5-7",
        "MP_N-500_H-7-3-5",
        "MP_N-500_H-5-7-3",

        "MP_N-500_L-01",
        "MP_N-500_L-02",
        "MP_N-500_L-04",
        "MP_N-500_L-05",

        "MP_N-500_M-01",
        "MP_N-500_M-03",
        "MP_N-500_M-04",
        "MP_N-500_M-05",

        "MP_N-500_E-5",
        "MP_N-500_E-10",
        "MP_N-500_E-15",
        "MP_N-500_E-25",
    ]

    options_list = [
        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "1"],       # DEFAULT

        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "3"],       # -H START
        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "5"],
        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "7"],
        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "3, 5"],
        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "5, 3"],
        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "3, 5, 7"],
        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "7, 3, 5"],
        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "5, 7, 3"], # -H END

        ["-L", "0.1", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "1"],       # -L START
        ["-L", "0.2", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "1"],
        ["-L", "0.4", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "1"],
        ["-L", "0.5", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "1"],       # -L END

        ["-L", "0.3", "-M", "0.1", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "1"],       # -M START
        ["-L", "0.3", "-M", "0.3", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "1"],
        ["-L", "0.3", "-M", "0.4", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "1"],
        ["-L", "0.3", "-M", "0.5", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "1"],       # -M END

        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "5", "-H", "1"],       # -E START
        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "10", "-H", "1"],
        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "15", "-H", "1"],
        ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "25", "-H", "1"],       # -E END
    ]

    for i in range(len(options_list)):
        start = time.time()
        print("Beginning iteration "+str(i)+": "+ file_names[i])

        # Use MultilayerPercepton and set options
        cls = Classifier(classname="weka.classifiers.functions.MultilayerPerceptron",
                         options=options_list[i])
        # Build classifier with train data
        cls.build_classifier(data)

        # Predictions stored in pout
        pout = PredictionOutput(
            classname="weka.classifiers.evaluation.output.prediction.PlainText")

        # Evaluate data on test data
        evaluation = Evaluation(data)
        evaluation.test_model(cls, test, output=pout)

        print(evaluation.summary())
        print(evaluation.class_details())
        print(evaluation.confusion_matrix)

        # Generate grid for ROC
        # plcls.plot_roc(evaluation, class_index=[0,1], wait=True)

        # mk dirs for output
        tempdir = dir / "Results/" / "MP-ALL_N-500_results/" / (file_names[i] + "_results/")
        tempdir.mkdir(parents=True, exist_ok=True)



        # Save summary, class details and confusion matrix to file
        result_output = file_names[i] + "_results.txt"
        print(tempdir)
        print(result_output)
        print((tempdir / result_output).absolute())
        output_eval(evaluation, tempdir / result_output)

        # Save the predicited results to file
        prediction_output = file_names[i] + "_prediction.txt"
        output_pred(pout, tempdir / prediction_output)

        end = time.time()
        timetaken = round(end - start,2)
        print("Time taken to run iteration "+str(i)+": %s seconds" % (timetaken))

    print("Multilayer Percepton complete")
