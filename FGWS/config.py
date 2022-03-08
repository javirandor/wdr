import argparse
import os


class Config:
    parser = argparse.ArgumentParser(description="args for experiments")
    parser.add_argument(
        "-mode",
        dest="mode",
        default="train",
        type=str,
        help="mode is either train, test, attack or detect",
    )
    parser.add_argument(
        "-model_type",
        dest="model_type",
        default="cnn",
        type=str,
        help="one of cnn, lstm, roberta",
    )
    parser.add_argument(
        "-model_name",
        dest="model_name",
        default="",
        type=str,
        help="name of the experiment",
    )
    parser.add_argument(
        "-attack",
        dest="attack",
        default="random",
        type=str,
        help="one of random, prioritized, genetic, pwws",
    )
    parser.add_argument(
        "-visible_devices",
        dest="visible_devices",
        default="0,1",
        type=str,
        help="which GPUs to use",
    )
    parser.add_argument(
        "-limit",
        dest="limit",
        default="0",
        type=int,
        help="truncates the respective dataset to a limited size",
    )
    parser.add_argument(
        "-num_epoch",
        dest="num_epoch",
        default="20",
        type=int,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "-dataset",
        dest="dataset",
        default="imdb",
        type=str,
        help="dataset to use; one of imdb, sst2",
    )
    parser.add_argument(
        "-fp_threshold",
        dest="fp_threshold",
        default="0.9",
        type=float,
        help="false positive threshold",
    )
    parser.add_argument(
        "-delta_thr",
        dest="delta_thr",
        default="None",
        type=str,
        help="delta threshold for detection",
    )
    parser.add_argument(
        "-gpu",
        dest="gpu",
        action="store_true",
        help="flag indicating whether to use GPU",
    )
    parser.add_argument(
        "-attack_train_set",
        dest="attack_train_set",
        action="store_true",
        help="whether to attack the train set",
    )
    parser.add_argument(
        "-attack_val_set",
        dest="attack_val_set",
        action="store_true",
        help="whether to attack the val set",
    )
    parser.add_argument(
        "-detect_val_set",
        dest="detect_val_set",
        action="store_true",
        help="whether to detect adversarial examples on the validation set",
    )
    parser.add_argument(
        "-test_on_val",
        dest="test_on_val",
        action="store_true",
        help="whether to test on validation set",
    )
    parser.add_argument(
        "-detect_baseline",
        dest="detect_baseline",
        action="store_true",
        help="whether to use baseline detection",
    )
    parser.add_argument(
        "-tune_delta_on_val",
        dest="tune_delta_on_val",
        action="store_true",
        help="use to tune delta on the validation set",
    )

    args = parser.parse_args()

    tune_delta_on_val = args.tune_delta_on_val
    detect_val_set = True if tune_delta_on_val else args.detect_val_set
    attack_val_set = args.attack_val_set
    test_on_val = args.test_on_val
    detect_baseline = args.detect_baseline
    fp_threshold = args.fp_threshold
    delta_thr = None if args.delta_thr == "None" else int(args.delta_thr)
    model_type = args.model_type
    attack = args.attack
    limit = args.limit
    gpu = args.gpu
    visible_devices = args.visible_devices
    mode = args.mode
    dataset = args.dataset
    attack_train_set = args.attack_train_set
    num_epoch = args.num_epoch
    model_name = args.model_name

    use_BERT = model_type == "roberta"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    project_root_path = "."
    data_root_path = "data"
    model_name_suffix = ""

    if mode == "train":
        model_name_suffix = "{}/{}".format(model_type, dataset)
    elif mode == "test":
        model_name_suffix = "test/{}/{}".format(model_type, dataset)
    elif mode == "attack":
        attack_set = (
            "train_set"
            if attack_train_set
            else "val_set"
            if attack_val_set
            else "test_set"
        )
        model_name_suffix = "limit_{}/{}/{}/{}/{}".format(
            limit, model_type, attack, dataset, attack_set
        )
    elif mode == "detect":
        attack_set = (
            "train_set"
            if attack_train_set
            else "val_set"
            if attack_val_set or tune_delta_on_val
            else "test_set"
        )

        model_name_suffix = "{}_attack_{}_{}_{}{}".format(
            attack,
            model_type,
            dataset,
            attack_set,
            "_limit_{}".format(int(limit)) if int(limit) > 0 else "",
        )

        model_name_suffix += "_fp_threshold_{}".format(fp_threshold)

        if delta_thr is not None:
            model_name_suffix += "_delta_thr_{}".format(delta_thr)

        if detect_baseline:
            model_name_suffix += "_detect_baseline"

    model_dir = (
        "models"
        if mode in ["train", "test"]
        else "attacks"
        if mode == "attack"
        else "detections"
    )
    model_base_path = "{}/{}/{}".format(data_root_path, model_dir, model_name_suffix)

    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path)

    tb_log_train_path = "{}/logs/tb_train".format(model_base_path)
    tb_log_val_path = "{}/logs/tb_val".format(model_base_path)

    if mode == "train":
        if not os.path.exists(tb_log_train_path):
            os.makedirs(tb_log_train_path)

        if not os.path.exists(tb_log_val_path):
            os.makedirs(tb_log_val_path)

    cf_path = "{}/data/pretrained/counter-fitted/counter-fitted-vectors.txt".format(
        data_root_path
    )
    glove_path = "{}/data/pretrained/gloVe/glove.42B.300d.txt".format(data_root_path)

    path_to_pre_trained_init = "{}/data/{}/pretrained_init.npy".format(
        data_root_path, dataset
    )

    keep_embeddings_fixed = False

    path_to_imdb = "{}/data/imdb".format(data_root_path)
    path_to_sst2 = "{}/data/sst2/tsv-format".format(data_root_path)

    seed_val = 42

    restore_model_path = "{}/models/{}/{}".format(data_root_path, model_type, dataset)
    load_model_path = "{}/checkpoints/e_best/model.pth".format(restore_model_path)

    val_split_size = 1000

    vocab_size = 0
    num_classes = 2
    max_len = 200 if dataset == "imdb" else 50
    bert_max_len = 256 if dataset == "imdb" else 128

    path_to_dist_mat = data_root_path + "/data/{}/dist_mat.npy".format(dataset)
    path_to_idx_to_dist_mat_idx_dict = (
        data_root_path + "/data/{}/idx_to_dist_mat_idx.pkl".format(dataset)
    )
    path_to_dist_mat_idx_to_idx_dict = (
        data_root_path + "/data/{}/dist_mat_idx_to_idx.pkl".format(dataset)
    )

    # Training params
    embed_size = 300
    batch_size_train = 100
    batch_size_val = 100
    batch_size_test = 100
    learning_rate = 1e-3
    dropout_rate = 0.1

    # Params for CNN
    filter_sizes = [2, 3, 4]
    stride = 1
    num_feature_maps = 100

    # Params for LSTM
    hidden_size = 128
    num_layers = 1

    # Params for RoBERTa
    if use_BERT:
        # RoBERTa params from https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md
        batch_size_train = 32 if dataset == "sst2" else 16
        batch_size_val = 32 if dataset == "sst2" else 16
        batch_size_test = 32 if dataset == "sst2" else 16
        weight_decay = 0.1
        warmup_percent = 0.06
        adam_eps = 1e-6
        learning_rate = 1e-5
        num_epoch = 10
        clip_norm = 0.0

    # Params for Genetic attack
    k = 8
    LM_cut = 4
    LM_window_size = 5
    max_alt_ratio = 0.2
    dist_metric = "euclidean"
    delta = 0.5 if dataset == "imdb" else 1.0
    num_pop = 60
    num_gen = 20
    path_to_attack_dist_mat = "{}/data/{}/attack_dist_mat_{}.npy".format(
        data_root_path, dataset, dist_metric
    )
    path_to_attack_dist_mat_word_to_idx = (
        "{}/data/{}/attack_dist_mat_word_to_idx.pkl".format(data_root_path, dataset)
    )
    path_to_attack_dist_mat_idx_to_word = (
        "{}/data/{}/attack_dist_mat_idx_to_word.pkl".format(data_root_path, dataset)
    )

    bootstrap_sample_size = 10000
    ci_alpha = 0.01

    pad_token = "<pad>"
    eos_token = "."
    unk_token = "<unk>"

    restore_delta_path = "{}/detections/delta_experiments/deltas/{}".format(
        data_root_path, "cont{}".format(fp_threshold)
    )

    if tune_delta_on_val and not os.path.exists(restore_delta_path):
        os.makedirs(restore_delta_path)

    restore_delta_path += "/{}_{}.pkl".format(model_type, dataset)
