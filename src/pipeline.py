import logging, os, random, json, time, zlib, glob
import pandas as pd
import fetch, config, baselines, build, evaluation
import diagnostic
from baselines import ExtractionConfig, DraftDataExtractor, BaselineEnricher, ProfileTuner

INTERMEDIATE_DIR = 'products'

def get_results_csv_path(cfg, topic):
    return cfg.paths.results_dir / f"{topic}.csv"

def get_estimated_effects_path(cfg, set_code, method_name, ext=None, split=None):
    base_dir = cfg.paths.results_dir / "baselines" / set_code
    if split is not None:
        base_dir = base_dir / f"split{split}"    
    if ext is not None:
        ext = "." + ext
        base_dir = base_dir / 'parts'
    else:
        ext = ""
    return  base_dir / f"{method_name}.{set_code}{ext}.csv"

def get_extractor_data_path(cfg, set_code, split=None, variant=None):
    sdesc = ""
    if variant is not None:
        sdesc += f'.{variant}'
    if split is not None:
        sdesc += f'.split{split}'    
    return cfg.paths.results_dir / INTERMEDIATE_DIR / set_code / f"extractor{sdesc}.dt.npz"

def get_hparams_profile_path(cfg, set_code, profile, seed):
    return cfg.paths.results_dir / INTERMEDIATE_DIR / set_code / "profiles" / profile / f"{seed}.json"


def build_extractor(cfg, set_code, split=None, extractor_class=DraftDataExtractor, 
                    pairs_df=None, extractor_variant=None):
    datapath = config.DataPath(cfg)
    if pairs_df is None:
        pairs_df = pd.read_csv(build.get_reference_effect_path(cfg, set_code, split=split))
    card_a_names=pairs_df['card_a'].unique().tolist()

    grouping = fetch.get_card_groups(datapath, set_code)
    metadata = fetch.get_card_metadata(datapath, set_code)

    extractor_data_path = get_extractor_data_path(cfg, set_code, split=split, variant=extractor_variant)
    logging.info(f"exractor path : {extractor_data_path}, {extractor_data_path.exists()}")
    if not extractor_data_path.exists():
        logging.info("extracting data")
        extractor = extractor_class(
            ExtractionConfig(set_code=set_code, split_rnd_seed=split), 
            metadata, grouping, card_a_names=card_a_names)
        draft_path, _ = fetch.get_draft_csv_path(datapath, set_code)
        extractor.process_drafts_file(draft_path)
        os.makedirs(os.path.dirname(extractor_data_path), exist_ok=True)
        extractor.to_file(extractor_data_path)
    else:
        logging.info(f"reading extracted features from {extractor_data_path}")
        extractor = DraftDataExtractor.from_file(extractor_data_path)
    return extractor



def build_baseline_estimates(cfg, set_code, method_names, idx=None, split=None, filter=None):

    logging.info(f"building baseline estimates {set_code} {method_names}")
    datapath = config.DataPath(cfg)
    pairs_df = pd.read_csv(build.get_reference_effect_path(cfg, set_code, split=split))
    ext = ""
    if idx is not None:
        card_a_list = sorted(list(pairs_df['card_a'].unique()))
        if filter is not None:
            card_a_list = [x for x in card_a_list if filter in x]
        if idx >= len(card_a_list):
            logging.info(f"idx {idx} out of range - we're done")
            return                
        card_a = card_a_list[idx]
        logging.info(f"subseting on card A: {card_a}")
        ext = f'{zlib.crc32(card_a.encode("utf-8")):08x}'
        pairs_df = pairs_df[pairs_df['card_a'] == card_a]

    #card_a_names=pairs_df['card_a'].unique().tolist()
    #grouping = fetch.get_card_groups(datapath, set_code)
    #metadata = fetch.get_card_metadata(datapath, set_code)
    
    
    extractor = build_extractor(cfg, set_code, split=split)
    DRAGONNET_NAMES = ['dragonnet', 'dragonnet_imp_att', 'dragonnet_cate_att']
    with_dragonnet = len([x for x in method_names if x in DRAGONNET_NAMES]) > 0
    method_names = [x for x in method_names if x not in DRAGONNET_NAMES]

    estimates = {}
    if method_names:
        logging.info(f"with methods {method_names}")
        enricher = BaselineEnricher(extractor, estimator_names=method_names)
        simp_estimates = enricher.get_estimates(pairs_df)
        estimates.update(simp_estimates)
    if with_dragonnet:
        logging.info("with dragonnet")
        import dragonnet
        enricher = dragonnet.DragonNetEnricher(extractor)
        drg_estimates = enricher.get_estimates(pairs_df)
        estimates.update(drg_estimates)        

    for method_name, est_df in estimates.items():
        dest_path = get_estimated_effects_path(cfg, set_code, method_name, ext=ext, split=split)
        logging.info(f"saving {method_name} => {dest_path}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        est_df.to_csv(dest_path)

def generate_seed():
    current_time = time.time_ns()
    pid = os.getpid()    
    raw_seed = current_time ^ (pid << 32)
    seed = raw_seed & ((1 << 32) - 1)    
    return seed

def hp_tuning_sample(cfg, set_code, profile, space_path):
    seed = generate_seed()
    dest_path = get_hparams_profile_path(cfg, set_code, profile, seed)    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    extractor = build_extractor(cfg, set_code)
    logging.info(f"tuning hyper-params for {set_code} {profile} {seed} => {dest_path}")
    #datapath = config.DataPath(cfg)
    pairs_df = pd.read_csv(build.get_reference_effect_path(cfg, set_code))
    card_a_names=pairs_df['card_a'].unique().tolist()
    tuner = ProfileTuner(extractor, card_a_names, seed=0)
    assert profile in ['logit', 'gbm_classifier', 'gbm']    
    logging.info(f"selected seed {seed}")

    if space_path is not None:
        with open(dest_path, "rt") as jf:
            hp_space = json.load(jf)
    else:
        hp_space = baselines.DEFAULT_SPACES[profile]
     
    params = baselines.random_grid(hp_space, 1, seed)[0]
    logging.info(f"generaed params {params}")
    logging.info(f"scoring")    
    result = tuner.score(profile, params)
    summary = {'profile': profile, 'seed': seed, 'params': params, **result}
    with open(dest_path, "wt") as jf:
        json.dump(summary, jf)
    

def summarize_results(cfg, topic, set_codes, method_names=[], split=None):
    mask_colnames = ["GSP","LD"]
    res = []
    for set_code in set_codes:
        df = pd.read_csv(build.get_reference_effect_path(cfg, set_code, split=split))
        
        ## col cleanup        
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if 'set_code' not in df.columns:
            df.insert(0, 'set_code', set_code)
        df = df[[x for x in df.columns if x not in mask_colnames]]
        
        if method_names:
            estimates = {}
            for method in method_names:
                ptrn_comp = str(get_estimated_effects_path(cfg, set_code, method, split=split))
                ptrn_parts = str(get_estimated_effects_path(cfg, set_code, method, ext='*', split=split))
                logging.info(f"{ptrn_comp} | {ptrn_parts}")
                fn_comp = glob.glob(ptrn_comp)
                fn_parts = glob.glob(ptrn_parts)
                logging.info(f"{len(fn_comp)}: {fn_comp}")
                logging.info(f" {len(fn_parts)} : {fn_parts}")                
                assert len(fn_comp) <= 1, "expecting none/single"
                assert (len(fn_comp)!=0) != (len(fn_parts) != 0), "expecting parts (x)or complete"

                if fn_comp:
                    fns = fn_comp
                else:
                    fns = fn_parts
                estimates[method] = pd.concat([pd.read_csv(x) for x in fns])
            df = evaluation.enrich_pairs_estimates(df, estimates, validate=True)
        res.append(df)

    cols = res[0].columns
    sdf = pd.concat([x[cols] for x in res], ignore_index=True)
    sdf.to_csv(get_results_csv_path(cfg, topic), index=False)
    
def independence_diagnostic(cfg, set_code):
    import ind_diagnostic as idiagnostic
    logging.info(f"diagnostic {set_code}")
    datapath = config.DataPath(cfg)
    pairs_df = pd.read_csv(build.get_reference_effect_path(cfg, set_code))[['card_a','group_b']]
    card_a_list = sorted(list(pairs_df['card_a'].unique()))
    grouping = fetch.get_card_groups(datapath, set_code)

    dest_path = cfg.paths.results_dir / INTERMEDIATE_DIR / set_code / "diagnostic" /  f"independence.csv"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    hbd = idiagnostic.HistoryBalanceDiagnostic(set_code, card_a_list, grouping)
    draft_path, _ = fetch.get_draft_csv_path(datapath, set_code)
    hbd.process_drafts_file(draft_path)
    res = hbd.get_balance_df()
    logging.info(f"saving => {dest_path}")
    res.to_csv(dest_path, index=False)


def instrument_diagnostic(cfg, set_code):
    logging.info(f"diagnostic {set_code}")
    datapath = config.DataPath(cfg)
    pairs_df = pd.read_csv(build.get_reference_effect_path(cfg, set_code))[['card_a','group_b']]
    card_a_list = sorted(list(pairs_df['card_a'].unique()))
    dest_path = cfg.paths.results_dir / INTERMEDIATE_DIR / set_code / "diagnostic" /  f"exclusion_diagnostic.csv"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    extractor = build_extractor(cfg, set_code)    
    diag = diagnostic.ExactMatchExclusionDiagnostic(extractor)
    res = diag.get_estimates(pairs_df)
    if 'set_code' not in res.columns:
        res.insert(0, 'set_code', set_code)
    logging.info(f"saving => {dest_path}")
    res.to_csv(dest_path, index=False)

def slot_diagnostic(cfg, set_code):
    logging.info(f"diagnostic {set_code}")    
    datapath = config.DataPath(cfg)
    metadata = fetch.get_card_metadata(datapath, set_code)
    grouping = fetch.get_card_groups(datapath, set_code)
    rares = [k for k,v in  metadata.items() if v['rarity'] in ['rare','mythic'] ]
    pickstats = fetch.get_card_picks(datapath, set_code)    
    card_a_list = ([x['card_name'] for x in pickstats if x['card_name'] in rares and  
                    x["p2p1_picked"] > 500 and x["p2p1_picked"] / x["p2p1_offered"] < 0.3])
    card_a_list = sorted(list(set(card_a_list)))
    outcomes = sorted(list(set(grouping.values())))
                         

    dest_path = cfg.paths.results_dir / INTERMEDIATE_DIR / set_code / "diagnostic" /  f"slot_diagnostic.csv"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    pairs_df = pd.DataFrame([
        dict(set_code=set_code, card_a=card_a, group_b=group_b)
        for card_a in card_a_list
        for group_b in outcomes
     ])

    extractor = build_extractor(cfg, set_code, pairs_df=pairs_df, extractor_variant='slots')    
    diag = diagnostic.ExactMatchExclusionDiagnostic(extractor)
    res = diag.get_estimates(pairs_df)
    if 'set_code' not in res.columns:
        res.insert(0, 'set_code', set_code)
    logging.info(f"saving => {dest_path}")
    res.to_csv(dest_path, index=False)

