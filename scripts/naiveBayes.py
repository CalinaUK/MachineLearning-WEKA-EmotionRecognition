from weka.classifiers import Classifier, Evaluation, PredictionOutput
import weka.plot.classifiers as plcls
from weka.core.classes import Random
from scripts.wekaloader import load_Arff_file
from helper import output_eval, output_pred

def run_naiveBayes(file):
    # Get filename from Pathlib object
    filename = file.parts[-1]
    dir = file.parents[0]

    print("Running NaiveBayes on %s" % filename)

    if not filename.endswith(".arff"):
        print("%s not ARFF file." % filename)
        return

    # Removes '.arff' from filename
    filename_base = filename[:-5]

    # Load data with class as first attr
    data = load_Arff_file(file)
    data.class_is_first()

    # Use BayesNet and set options
    cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
    # Predictions stored in pout
    pout = PredictionOutput(
        classname="weka.classifiers.evaluation.output.prediction.PlainText")

    # Evaluate data
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(cls, data, 10, Random(1), output=pout)

    print(evaluation.summary())
    print(evaluation.class_details())
    print(evaluation.confusion_matrix)

    # Generate grid for ROC
    # plcls.plot_roc(evaluation, class_index=[0,1], wait=True)

    # mk dirs for output
    dir = dir / "naiveBayes_results"
    dir.mkdir(parents=True, exist_ok=True)

    # Save summary, class details and confusion matrix to file
    result_output = filename_base + "_naiveBayes_eval_results.txt"
    output_eval(evaluation, dir / result_output)

    # Save the predicited results to file
    prediction_output = filename_base + "_naiveBayes_pred_results.txt"
    output_pred(pout, dir / prediction_output)

    print("BayesNet complete")
