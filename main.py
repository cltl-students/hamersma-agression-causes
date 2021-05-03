from preprocessing import preprocess
from approach1_rulebased import get_predictions_rulebased
from approach2_machine_learning import get_predictions_ml
from bertopic_clustering import cluster_precursors

def main():
    '''Main function to use from commandline, preprocess input to generate embeddings, detect agression clauses using
    provided approach, extract features and labels from training and features from input data, trains a model and
    classifies test data using the trained model, evaluates predictions and goldlabels from input'''
    inputfile = 'sample_input.xls'
    preprocess(inputfile)
    get_predictions_rulebased()
    get_predictions_ml()

    ### only clusters with enough data, else everything in outlier cluster
    cluster_precursors()

if __name__ == '__main__':
    main()