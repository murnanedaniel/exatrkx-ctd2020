import argparse
import yaml

def parse_args():
    
    '''
    This pattern allows defaults to be set and overridden by a config file, and then the command-line.
    '''
    
    # Handle required arguments
    conf_parser = argparse.ArgumentParser('pipeline.py', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    add_arg = conf_parser.add_argument
    add_arg('config_file', nargs='?', default=None)
    args, remaining_args = conf_parser.parse_known_args()
    
    # Set defaults (maybe these themselves could be in a default config file in the utils folder??)
    defaults = {'stage': 'seed',
                'force': None, 
                'task': 0,
                'n_tasks': 1,
                'pt_cut': 0,
                'doublet_threshold': 0,
                'nb_to_process': -1,
                'feature_names': ['x','y','z','cell_count', 'cell_val',
                     'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi'],
                'rank': 0,
                'n_ranks': 1,
                'verbose': False,
                'resume': False
                }
    
    if args.config_file is None:
        print("No config file provided. Setting default to be: Seeding/src/configs/seed.yaml")
        try:
            with open(args.config_file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                defaults.update(config)
        except:
            print("But default config file is not present. Create one to get started and include it when running the pipeline.")
            raise
    else:
        try:
            with open(args.config_file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                defaults.update(config)
        except:
            print("Config file that was provided does not exist.")
            raise
    
    
    # Handle command-line changes to default arguments
    parser = argparse.ArgumentParser()
    parser.set_defaults(**defaults)
    add_arg = parser.add_argument

    add_arg('stage', help="Which stage to run.", choices=['seed', 'label', 'train', 'preprocess', 'build_doublets', 'build_triplets', 'train_embedding', 'train_filter', 'train_doublets', 'train_triplets'], nargs='?')
    add_arg('--force', help="Specify stage to force reprocessing. All subsequent stages will be reprocessed.", nargs='?', const='all')
    add_arg('--nb-to-preprocess', help="How many subfolders of data", type=int)
    
    add_arg('--include-endcaps', help='Include endcaps', action='store_true') #The logic here is a little counter-intuitive to get the default to be endcaps removed (ditto noise)
    add_arg('--include-noise', help='Include noise', action='store_true')
    
    add_arg('--pt-cut', help='Transverse momentum, below which tracks are excluded', type=float)
    add_arg('--task', help='Which GPU number is this script running on', type=int, default=0)
    add_arg('--n_tasks', help='Total number of GPUs available', type=int, default=1)
    
    add_arg('--distributed', choices=['ddp-file', 'ddp-mpi', 'cray'])
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--rank-gpu', action='store_true')
    add_arg('--ranks-per-node', default=8)
    add_arg('--gpu', type=int)
    add_arg('--fom', default=None, choices=['last', 'best'],
            help='Print figure of merit for HPO/PBT')
    add_arg('--interactive', action='store_true')
        
    return parser.parse_args(remaining_args)