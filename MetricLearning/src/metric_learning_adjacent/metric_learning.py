import os

from .preprocess import preprocess_stage_1, preprocess_stage_2
from .train_embed import train_embed
from .train_filter import train_filter

def main(experiment_name,
         data_storage,
         artifact_storage,
         feature_names,
         preprocess_data_path,
         filter_nb_layer=4,
         filter_nb_hidden=2048,
         force=False):

    print("\nStarting metric learning.")
    data_storage = os.path.join(data_storage, 'metric_learning')

    # preprocess for embed model
    stage_1_data_path = preprocess_stage_1.preprocess(experiment_name,
                                                      artifact_storage,
                                                      preprocess_data_path,
                                                      feature_names,
                                                      data_storage+'/stage_1',
                                                      nb_train = 10**7,
                                                      nb_valid = 5*10**6,
                                                      nb_test  = 5*10**6,
                                                      force=force)
    # train embed model
    emb_model = train_embed.main(experiment_name,
                                 stage_1_data_path,
                                 artifact_storage,
                                 force=force)

    # preprocess for filter model
    stage_2_data_path = preprocess_stage_2.preprocess(experiment_name,
                                                      artifact_storage,
                                                      preprocess_data_path,
                                                      feature_names,
                                                      emb_model,
                                                      data_storage+'/stage_2',
                                                      nb_train = 4*10**7,
                                                      nb_valid = 2*10**6,
                                                      nb_test  = 2*10**6,
                                                      force=force)

    # train filter model
    filter_model = train_filter.main(experiment_name,
                                     stage_2_data_path,
                                     artifact_storage,
                                     nb_layer=filter_nb_layer,
                                     nb_hidden=filter_nb_hidden,
                                     force=force)
    return emb_model, filter_model
