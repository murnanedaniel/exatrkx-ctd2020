# SYSTEM
import os
import sys
import logging

# EXTERNALS
import yaml

# UTILS
from utils.pipeline_utils import parse_args

# STAGES OF PIPELINE
sys.path.append('.')
from MetricLearning.src.preprocess_with_dir import preprocess
from MetricLearning.src.metric_learning_adjacent import build_graphs as build_doublet_graphs
from GraphLearning import build_triplets
from Seeding import seed
from Labelling import label


    
def build(args):
    
    # Handle the logic flow of forced data re-building (i.e. ignoring existing data)
    
    force_order = {"all": 0, "preprocess": 1, "build_doublets": 2, "build_triplets": 3, "seed": 4, "label": 5}
    force = [False]*len(force_order)
    if args.force is not None:
        force[force_order[args.force]:] = [True]*(len(force_order)-force_order[args.force])
#     print(force)
    
    # BUILD / INFERENCE PIPELINE
    # -------------------------------

    
    ## PREPROCESS DATA - Calculate cell features, select barrel, no noise, construct event-by-event files
    # Checks for existence of preprocessed dataset. 
    
    print('--------------------- \n Preprocessing... \n--------------------')
    
    args.data_storage_path = os.path.join(args.data_storage, args.name) # Could move these definitions into preprocess.main
#     args.artifact_storage_path = os.path.join(args.metric_artifacts, args.name)
    preprocess_data_path, feature_names = preprocess.main(args, force=force[force_order["preprocess"]])
    if args.stage == 'preprocess': return

    
    ## BUILD DOUBLET GRAPHS - Apply embedding, run filter inference, post-process, construct event doublet files
    # Checks for existence of doublet graphs. 
    
    print('--------------------- \n Building doublets... \n--------------------')
    
    build_doublet_graphs.main(args, force=force[force_order["build_doublets"]])    
    if args.stage == 'build_doublets': return
    
    
    ## BUILD TRIPLET GRAPHS - Apply doublet GNN model to doublet graphs for classification, construct triplets using score & cut
    # Checks for existence of triplet graphs
    
    print('--------------------- \n Building triplets... \n--------------------')
    
    build_triplets.main(args, force=force[force_order["build_triplets"]])
    if args.stage == 'build_triplets': return
    
    
    ## CLASSIFY TRIPLET GRAPHS FOR SEEDS - Classify, apply cut to triplet scores, print out into specified format
    # Checks for existence of seed dataset
    if args.stage == "seed":
        print('--------------------- \n Seeding... \n--------------------')
        seed.main(args, force=force[force_order["seed"]])
     
        return
    
    
    ## CLASSIFY TRIPLET GRAPHS FOR TRACK LABELS - Classify, convert to doublets, construct sparse undirected matrix, run DBSCAN, print out into specified format
    
    if args.stage == "label":
        print('--------------------- \n Labelling... \n--------------------')
        label.main(args, force=force[force_order["label"]])
        
        
        return


def train(args):
    
    # TRAIN PIPELINE
    # ------------------------------

    ## PREPROCESS DATA - As above

    
    if args.stage == 'build_triplets': return
    ## TRAIN EMBEDDING

    
    if args.stage == 'build_triplets': return
    ## BUILD EMBEDDED SPACE

    
    if args.stage == 'build_triplets': return
    ## TRAIN FILTER
    
    if args.stage == 'build_triplets': return

    ## BUILD DOUBLET GRAPHS

    
    if args.stage == 'build_triplets': return
    ## TRAIN DOUBLET GNN

    
    if args.stage == 'build_triplets': return
    ## BUILD TRIPLET GRAPHS

    
    if args.stage == 'build_triplets': return
    ## TRAIN TRIPLET GNN






if __name__ == "__main__":
    
    args = parse_args()
    
    if args.stage in ["seed", "label", "preprocess", "build_doublets", "build_triplets"]:
        build(args)
    
    elif args.stage in ["train", "train_embedding", "train_doublets", "train_triplets"]:
        train(args)
    