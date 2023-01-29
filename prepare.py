import os
import random
import json
import logging
from datasets.prepare_EMOVDB import prepare_emovdb
from datasets.prepare_ESD import prepare_esd
from datasets.prepare_IEMOCAP import prepare_iemocap
from datasets.prepare_JLCORPUS import prepare_jlcorpus
from datasets.prepare_RAVDESS import prepare_ravdess
from utils.utils import get_labels

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def prepare_train(
    save_json_train,
    save_json_valid,
    save_json_test = None,
    split_ratio=[90, 10],
    win_len=0.2,
    stride=0.2,
    seed=12,
    emovdb_folder=None,
    esd_folder=None,
    iemocap_folder=None,
    jlcorpus_folder=None,
    ravdess_folder=None,
):
    # setting seeds for reproducible code.
    random.seed(seed)

    if os.path.exists(save_json_train) and os.path.exists(save_json_valid):
        logger.info("train/valid json both exist, skipping preparation.")
        return
    
    all_dict = {}
    if emovdb_folder is not None:
        if not os.path.exists(emovdb_folder + "EMOV-DB.json"):
            emovdb = prepare_emovdb(emovdb_folder, emovdb_folder + "EMOV-DB.json", seed)
        else:
            logger.info(f"{emovdb_folder}EMOV-DB.json exists, skipping EMOV-DB preparation.")
            with open(f'{emovdb_folder}EMOV-DB.json', 'r') as f:
                emovdb = json.load(f)
        all_dict.update(emovdb.items())
    else:
        logger.info("EMOV-DB is not used in this exp.")
        
    if esd_folder is not None:
        if not os.path.exists(esd_folder + "ESD.json"):
            esd = prepare_esd(esd_folder, esd_folder + "ESD.json", seed)
        else:
            logger.info(f"{esd_folder}ESD.json exists, skipping ESD preparation.")
            with open(f'{esd_folder}ESD.json', 'r') as f:
                esd = json.load(f)
        all_dict.update(esd.items())
    else:
        logger.info("ESD is not used in this exp.")
        
    if iemocap_folder is not None:
        if not os.path.exists(iemocap_folder + "IEMOCAP.json"):
            iemocap = prepare_iemocap(iemocap_folder, iemocap_folder + "IEMOCAP.json", seed)
        else:
            logger.info(f"{iemocap_folder}IEMOCAP.json exists, skipping IEMOCAP preparation.")
            with open(f'{iemocap_folder}IEMOCAP.json', 'r') as f:
                iemocap = json.load(f)
        all_dict.update(iemocap.items())
    else:
        logger.info("IEMOCAP is not used in this exp.")
        
    if jlcorpus_folder is not None:
        if not os.path.exists(jlcorpus_folder + "JL_CORPUS.json"):
            jlcorpus = prepare_jlcorpus(jlcorpus_folder, jlcorpus_folder + "JL_CORPUS.json", seed)
        else:
            logger.info(f"{jlcorpus_folder}JL_CORPUS.json exists, skipping JL_CORPUS preparation.")
            with open(f'{jlcorpus_folder}JL_CORPUS.json', 'r') as f:
                jlcorpus = json.load(f)
        all_dict.update(jlcorpus.items())
    else:
        logger.info("JL_CORPUS is not used in this exp.")
        
    if ravdess_folder is not None:
        if not os.path.exists(ravdess_folder + "RAVDESS.json"):
            ravdess = prepare_ravdess(ravdess_folder, ravdess_folder + "RAVDESS.json", seed)
        else:
            logger.info(f"{ravdess_folder}RAVDESS.json exists, skipping RAVDESS preparation.")
            with open(f'{ravdess_folder}RAVDESS.json', 'r') as f:
                ravdess = json.load(f)
        all_dict.update(ravdess.items())
    else:
        logger.info("RAVDESS is not used in this exp.")
    
    bad_keys = []
    for key in all_dict.keys():
        try:
            intervals, ctc_label, frame_label = get_labels(all_dict[key], win_len, stride)
            all_dict[key]["frame_label"] = frame_label
            all_dict[key]["ctc_label"] = ctc_label
        except:
            logger.info(f"Impossible to get labels for id {key}, window too large.")
            bad_keys.append(key)
            continue
    for key in bad_keys:
        del all_dict[key]

    data_split = split_sets(all_dict, split_ratio)
    
    train_ids = data_split["train"]
    train_split = {}
    for id in train_ids:
        train_split[id] = all_dict[id]
    
    valid_ids = data_split["valid"]
    valid_split = {}
    for id in valid_ids:
        valid_split[id] = all_dict[id]

    create_json(train_split, save_json_train)
    create_json(valid_split, save_json_valid)
    if save_json_test is not None:
        test_ids = data_split["test"]
        test_split = {}
        for id in test_ids:
            test_split[id] = all_dict[id]
        create_json(test_split, save_json_test)

def prepare_test(
    ZED_folder,
    save_json_test,
    win_len,
    stride,
):  
    if os.path.exists(save_json_test):
        logger.info("test json exists, skipping preparation.")
        return

    try:
        with open(f'{ZED_folder}/ZED.json', 'r') as f:
            all_dict = json.load(f)
    except:
        logger.info(f"ZED.json can't be found under {ZED_folder}")
        return

    bad_keys = []
    for key in all_dict.keys():
        try:
            all_dict[key]["wav"] = all_dict[key]["wav"].replace("datafolder", ZED_folder)
            intervals, ctc_label, frame_label = get_labels(all_dict[key], win_len, stride)
            all_dict[key]["frame_label"] = frame_label
            all_dict[key]["ctc_label"] = ctc_label
        except:
            logger.info(f"Impossible to get labels for id {key}, window too large.")
            bad_keys.append(key)
            continue
    for key in bad_keys:
        del all_dict[key]
    
    create_json(all_dict, save_json_test)


def split_sets(data_dict, split_ratio, splits=["train", "valid"]):
    """Randomly splits the wav list into training, validation, and test lists.
    Arguments
    ---------
    data_dict : list
        a dictionary of id and its corresponding audio information
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respectively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.
    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    assert len(splits) == len(split_ratio)
    id_list = list(data_dict.keys())
    # Random shuffle of the list
    random.shuffle(id_list)
    tot_split = sum(split_ratio)
    tot_snts = len(id_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = id_list[0:n_snts]
        del id_list[0:n_snts]
    if len(split_ratio) == 3:
        data_split["test"] = id_list
    return data_split


def create_json(data, json_file):
    """
    Creates the json file given a list of wav information.
    Arguments
    ---------
    data : dict
        The dict of wav information (path, label, gender).
    json_file : str
        The path of the output json file
    """
    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(data, json_f, indent=2)
    logger.info(f"{json_file} successfully created!")
