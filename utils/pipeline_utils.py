import argparse
import yaml

def parse_args():
    
    '''
    This pattern allows defaults to be set and overridden by a config file, and then the command-line.
    '''
    
    # Handle required arguments
    conf_parser = argparse.ArgumentParser('pipeline.py', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    add_arg = conf_parser.add_argument
    add_arg('config_file', nargs='?', default='Seeding/src/configs/seed.yaml')
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
                     'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']}
    
    # Handle config file changes to default arguments
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config is not None:
        defaults.update(config)
    
    # Handle command-line changes to default arguments
    parser = argparse.ArgumentParser(parents=[conf_parser])
    parser.set_defaults(**defaults)
    add_arg = parser.add_argument
    
    add_arg('stage', help="Which stage to run.", choices=['seed', 'label', 'train', 'preprocess', 'build_doublets', 'build_triplets', 'train_embedding', 'train_doublets', 'train_triplets'], nargs='?')
    add_arg('--force', help="Specify stage to force reprocessing. All subsequent stages will be reprocessed.", nargs='?', const='all')
    add_arg('--nb-to-preprocess', help="How many subfolders of data", type=int)
    
    add_arg('--include-endcaps', help='Include endcaps', action='store_true') #The logic here is a little counter-intuitive to get the default to be endcaps removed (ditto noise)
    add_arg('--include-noise', help='Include noise', action='store_true')
    
    add_arg('--pt-cut', help='Transverse momentum, below which tracks are excluded', type=float, default=0)
    add_arg('--task', help='Which GPU number is this script running on', type=int, default=0)
    add_arg('--n_tasks', help='Total number of GPUs available', type=int, default=1)
    
    return parser.parse_args(remaining_args)