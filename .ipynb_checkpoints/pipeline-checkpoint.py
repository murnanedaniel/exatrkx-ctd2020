# IMPORTS


# READ CONFIGS

def parse_args():
    parser = argparse.ArgumentParser('pipeline.py')

# BUILD / INFERENCE PIPELINE
# -------------------------------

## PREPROCESS DATA - Calculate cell features, select barrel, no noise, construct event-by-event files

## BUILD DOUBLET GRAPHS - Apply embedding, run filter inference, post-process, construct event doublet files

## BUILD TRIPLET GRAPHS - Apply doublet GNN model to doublet graphs for classification, construct triplets using score & cut

## CLASSIFY TRIPLET GRAPHS FOR SEEDS - Classify, apply cut to triplet scores, print out into specified format

## CLASSIFY TRIPLET GRAPHS FOR TRACK LABELS - Classify, convert to doublets, construct sparse undirected matrix, run DBSCAN, print out into specified format


# TRAIN PIPELINE
# ------------------------------






if __name__ == "__main__":
    