import os, logging
from config import DataPath
import pandas as pd
import fetch
import ivexp


def get_reference_effect_path(cfg, set_code, split=None):
    sdesc = ""
    if split is not None:
        sdesc = f'.split{split}'
    return cfg.paths.results_dir / f"ref{sdesc}.{set_code}.csv"

def build_reference_effects(cfg, set_code, bomb_ata_threshold=3.0, split=None):
    dest_path = get_reference_effect_path(cfg, set_code, split=split)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logging.info(f"constructing reference-effect for {set_code} at {dest_path}")
    datapath = DataPath(cfg)
    bomb_names = get_most_picked_cards(datapath, set_code)    
    logging.info(f"bomb-cards: [{len(bomb_names)}]: {bomb_names}")
    assert (len(bomb_names) > 0)
    
    grouping = fetch.get_card_groups(datapath, set_code)
    metadata = fetch.get_card_metadata(datapath, set_code)
    
    pce = ivexp.PackCausalExperiment(set_code, bomb_names,  grouping, split_rnd_seed=split)

    draft_csv_path, exists  = fetch.get_draft_csv_path(datapath, set_code)
    assert exists
    logging.info(f"processing {draft_csv_path}")
    pce.process_drafts_file(draft_csv_path)
    logging.info("done processing")
    refdf = pce.get_single_pack_df(1)
    logging.info(f"produced {refdf.shape} {refdf.columns}")
    
    refdf.to_csv(dest_path)

    if 'set_code' not in refdf.columns:
        refdf.insert(0, 'set_code', set_code)    
    refdf.to_csv(dest_path)


def get_most_picked_cards(datapath, set_code, pick_threshold=4.0, count_threshold=500):
    picks = pd.DataFrame(fetch.get_card_picks(datapath, set_code))
    pref = "p2p1"
    fpdf = picks[(picks[f"{pref}_offered"] / picks[f"{pref}_picked"] < pick_threshold) & 
                (picks[f"{pref}_picked"] > count_threshold)]
    names = list(fpdf["card_name"])
    return names

