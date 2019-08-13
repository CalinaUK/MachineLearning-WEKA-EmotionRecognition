from weka.clusterers import Clusterer, ClusterEvaluation
from scripts.wekaloader import load_Arff_file
import weka.plot.clusterers as plc
from weka.core.converters import Instances
from helper import output_cluster

def run_clusterer(file):
    # Get filename from Pathlib object
    filename = file.parts[-1]
    dir = file.parents[0]

    print("Running Clusterer on %s" % filename)

    if not filename.endswith(".arff"):
        print("%s not ARFF file." % filename)
        return

    # Removes '.arff' from filename
    filename_base = filename[:-5]

    # Load data with class as first attr
    full = load_Arff_file(file)
    full.class_is_first()

    full_withoutclass = load_Arff_file(file)
    #data.delete_first_attribute()

    data = Instances.copy_instances(full)
    data.no_class()
    data.delete_first_attribute()

    dir = dir / "cluster_results_optimum"
    dir.mkdir(parents=True, exist_ok=True)
    # Init clusterer

    #"-N", "-1",
    n = "2"


    if (filename_base.startswith("fer2018_")):
        print("Changing number of clusters to 7")
        n = "7"


	#clusterer = Clusterer(classname="weka.clusterers.EM", options=[ "-S", "10", "-N", n])   
	#clusterer = Clusterer(classname="weka.clusterers.FarthestFirst", options=[ "-S", "10", "-N", n])
    clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=[ "-S", "10", "-N", n])
    clusterer.build_clusterer(data)

    evaluation = ClusterEvaluation()
    evaluation.set_model(clusterer)
    evaluation.test_model(full)

    str1 = str(filename_base) + "_cl_res.txt"

    output_results = dir / str1
    output_cluster(evaluation, output_results)



    #print("Classes to cluster: " + str(evaluation.classes_to_clusters))


    #START
    #clusterer_general = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", n, "-S", "10"])

    #clusterer_general.build_clusterer(full_withoutclass)

    #evaluation_general = ClusterEvaluation()
    #evaluation_general.set_model(clusterer_general)
    #evaluation_general.test_model(full_withoutclass)

    #clusterer_classremoved = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", n, "-S", "10"])

    #clusterer_classremoved.build_clusterer(data)

    #evaluation_classremoved = ClusterEvaluation()
    #evaluation_classremoved.set_model(clusterer_classremoved)
    #evaluation_classremoved.test_model(data)

    #END
    #print("# clusters: " + str(evaluation.num_clusters))
    #print("log likelihood: " + str(evaluation.log_likelihood))
    #print("Cluster results: " + str(evaluation.cluster_results))

    #print("cluster assignments:\n" + str(evaluation.cluster_assignments))
    #plc.plot_cluster_assignments(evaluation, data, inst_no=True)

    #print("CLUSTER RESULTS")
    #eval = evaluation.cluster_results
    #print("CLASSES TO CLUSTER")
    #print(evaluation.classes_to_clusters)


    #cluster the data
    #for inst in data:
    #    cl = clusterer.cluster_instance(inst) # 0-based cluster index
    #    dist = clusterer.distribution_for_instance(inst)    # cluster membership distribution
    #    print("cluster = %s"  %str(cl))
    #    print("distribution = %s" % str(dist))

    # mk dirs for output


    #output_results_general = dir / "cl_res_gen.txt"
    #output_cluster(evaluation_general, output_results_general)

    #output_results_classremoved = dir / "cl_res_classremoved.txt"
    #output_cluster(evaluation_classremoved, output_results_classremoved)
