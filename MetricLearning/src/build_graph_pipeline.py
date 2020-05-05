import os
import sys
import logging
import argparse

from preprocess_with_dir import preprocess
from metric_learning_adjacent import embed_hits, metric_learning, build_graphs

def read_args():
    parser = argparse.ArgumentParser(description='Args for preprocessing data')
    add_arg = parser.add_argument
    
    add_arg('-v', '--verbose', action='store_true')

    add_arg('--raw_data_path', help='Path to raw data', required=True)
    add_arg('--detector_path', help='Path to detectors.csv', required=True)

    add_arg('--name', help='Experiment name', required=True)
    add_arg('--data_storage', help='Path to store intermediate data', required=True)
    add_arg('--artifact_storage', help='Path to training artifacts (models, logs, etc.)', required=True)

    add_arg('--include-endcaps', help='Include endcaps', action='store_false') #The logic here is a little reversed to get the default to be endcaps removed (ditto noise)
    add_arg('--include-noise', help='Include noise', action='store_false')
    add_arg('--nb-to-preprocess', help="How many subfolders of data", type=int, default=-1)
    
    add_arg('--pt-cut', type=float, default=0)
    add_arg('--task', help='Which GPU number is this script running on', type=int, default=0)
    add_arg('--n_tasks', help='Total number of GPUs available', type=int, default=1)

    add_arg('--operation', help='Use to specify if particular part of pipeline to run', default='all')
    add_arg('--force_pre', help='Forces preprocessing', action='store_true')
    add_arg('--force_train', help='Forces training', action='store_true')

    return parser.parse_args()

def main():
    args = read_args()
    args.data_storage = os.path.join(args.data_storage, args.name)
    args.artifact_storage = os.path.join(args.artifact_storage, args.name)

    # PREPROCESS - Parse TrackML files for feature learning
    if args.operation in ['preprocess', 'train', 'build', 'all']:
        preprocess_data_path, feature_names = preprocess.main(args.raw_data_path,
                                                          args.data_storage,
                                                          args.detector_path,
                                                          args.name,
                                                          args.include_endcaps,
                                                          args.include_noise,
                                                          args.nb_to_preprocess,
                                                          args.pt_cut,
                                                          args.task,
                                                          args.n_tasks,
                                                          args.force_pre,
                                                          args.verbose)


    # METRIC LEARNING - Train embedding and filtering models
    if args.operation in ['train', 'build', 'all']:
        emb_model, filter_model = metric_learning.main(args.name,
                                                   args.data_storage,
                                                   args.artifact_storage,
                                                   feature_names,
                                                   preprocess_data_path,
                                                   filter_nb_layer=3,
                                                   filter_nb_hidden=512,
                                                   force=args.force_train)

    # BUILD GRAPHS - Apply embedding+filtering models to new data and build hitgraphs
    if args.operation in ['build', 'all']:
        build_graphs.main(args.name,
                      args.data_storage,
                      feature_names,
                      preprocess_data_path,
                      emb_model,
                      filter_model,
                      emb_radius = 0.4,
                      filter_threshold = 0.95,
                      task = args.task,
                      n_tasks = args.n_tasks)

    # POSTPROCESS - Prepare hitgraphs for doublet classification applying pT cut, graph splitting, etc.
    if args.operation in ['postprocess', 'all']:
        postprocess.main(args.name,
                        args.data_storage,
                        args.pt_min,
                        args.num_phi_sections,
                        args.threshold,
                        args.embed_feats,
                        args.augmented)

        
        
if __name__ == "__main__":
    main()
