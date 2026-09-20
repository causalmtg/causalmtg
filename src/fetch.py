
import os, time, json
from datetime import datetime
import requests
import gzip, logging
import shutil
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from cachetools import cached, LRUCache
from pathlib import Path
from typing import Tuple
# Configuration
BASE_DRAFT_URL_17LANDS = "https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/"
BASE_DRAFT_URL = "https://huggingface.co/datasets/causalmtg/causalmtg/resolve/main/drafts/"

DRAFT_TYPE_URLS = {
    'DSK' : 'draft_data_public.DSK.PremierDraft.csv.gz',
    'MKM' : 'draft_data_public.MKM.PremierDraft.csv.gz',
    'DFT' : 'draft_data_public.DFT.PremierDraft.csv.gz',
    'BLB' : 'draft_data_public.BLB.PremierDraft.csv.gz',
    ## 'FIN' : 'draft_data_public.FIN.PremierDraft.csv.gz',    
}

SET_CODE_EXTENSIONS = {}
DRAFTS_DIR = "drafts"
METADATA_DIR = "metadata"

def get_draft_csv_path(cfg, set_code, type_urls=DRAFT_TYPE_URLS) -> Tuple[Path, bool]:
    filename_gz = type_urls[set_code]
    target_filename = filename_gz.replace('.gz', '')
    return cfg.get_path(Path(DRAFTS_DIR) / target_filename)
    
def download_and_extract_draft_dataset(cfg, set_code, type_urls=DRAFT_TYPE_URLS, **kwargs):

    output_path, exists = get_draft_csv_path(cfg, set_code, **kwargs)
    if exists: 
        return output_path, exists
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if file exists
    if os.path.exists(output_path):
        print(f"Skipping '{set_code}': File already exists at {output_path}")
        return
            
    logging.info(f"Downloading '{set_code}' from 17Lands...")
    filename_gz = type_urls[set_code]
    download_url = os.path.join(BASE_DRAFT_URL, filename_gz)

    # Stream the download
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
                
        # Decompress and write the file
        with gzip.GzipFile(fileobj=r.raw) as gz:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(gz, f_out)
                        
    logging.info(f"Successfully downloaded and extracted to: {output_path}")
            


def download_and_extract_datasets(cfg,  type_urls=DRAFT_TYPE_URLS, **kwargs):
    """
    Downloads and extracts .csv.gz files from 17Lands.
    Only downloads if the target .csv file does not already exist.
    """
    # Ensure the target directory exists
    
    for set_code, filename_gz in type_urls.items():
        try:
            download_and_extract_draft_dataset(cfg, set_code, type_urls=type_urls,**kwargs)            
        except Exception as e:
            logging.exception(f"Failed to download {set_code}: {e}")

def build_card_raw_metadata(cfg, set_code):
    """
    Fetches all cards for a specific set from Scryfall and formats 
    them as 'Color_Type' (e.g., 'UB_Creature', 'R_Instant', 'C_Artifact').
    'C' is used for Colorless.
    """
    print(f"Fetching card metadata for set [{set_code.upper()}] from Scryfall...")
    url = f"https://api.scryfall.com/cards/search?q=set:{set_code}"
    metadata = {}
    
    headers = {'User-Agent': 'MTGCausalExperiment/1.0'}
    res = []
    while url:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Scryfall API Error {response.status_code}")
            break
            
        json_data = response.json()
        res += (json_data.get('data',[]))
        url = json_data.get('next_page')
        if url: time.sleep(0.1)
    res = {x["name"] : x for x in res}
    return res

def build_card_raw_metadata_ext(cfg, set_code):
    set_code = set_code.upper()
    res = {}
    set_codes = [set_code] + SET_CODE_EXTENSIONS.get(set_code,[])
    logging.info(f"set_codes: {set_codes}")
    for x in set_codes:
        res.update(build_card_raw_metadata(cfg, x))
    return res

def build_card_groups(cfg, set_code):
    """
    Fetches all cards for a specific set from Scryfall and formats 
    them as 'Color_Type' (e.g., 'UB_Creature', 'R_Instant', 'C_Artifact').
    'C' is used for Colorless.
    """
    set_code = set_code.upper()
    raw = get_card_metadata(cfg, set_code) 
    metadata = {}
    for card in raw.values():
        name = card.get('name')
        if 'colors' in card: colors = card['colors']
        elif 'card_faces' in card and 'colors' in card['card_faces'][0]: colors = card['card_faces'][0]['colors']
        else: colors = []
        color_str = "".join(colors) if colors else "C"
        type_line = card.get('type_line', '')
        primary_type = "Other"
        for t in ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Land", "Planeswalker"]:
            if t in type_line:
                primary_type = t
                break
        metadata[name] = f"{color_str}_{primary_type}"
    return metadata

def get_card_groups_path(cfg, set_code):
    return  cfg.get_path(os.path.join(METADATA_DIR, f"groups.{set_code}.json"))

def get_card_metadata_path(cfg, set_code):
    return  cfg.get_path(os.path.join(METADATA_DIR, f"metadata.{set_code}.json"))

def get_card_picks_path(cfg, set_code):
    return  cfg.get_path(os.path.join(METADATA_DIR, f"picks.{set_code}.json"))

@cached(LRUCache(maxsize=20))
def get_card_metadata_i(cfg, set_code, desc, download, path_fn, build_fn):
    dest_path, exists = path_fn(cfg, set_code)
    if exists:
        with open(dest_path, "rt") as mf:
            logging.info(f"reading {desc} at {dest_path}")
            return json.load(mf)
    if not download:
        raise Exception(f"No metadata")
    
    metadata = build_fn(cfg, set_code)
    print(f"writing {desc} at {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wt") as mf:
            json.dump(metadata, mf)
    return metadata

def build_card_picks(cfg, set_code, format_type="PremierDraft"):
    draft_path, exists = get_draft_csv_path(cfg, set_code)
    assert exists
    cps = CardPickStats() 
    cps.process_drafts_file(draft_path)
    return cps.get_stats()

def get_card_groups(cfg, set_code, download=True):
    return get_card_metadata_i(cfg, set_code, desc='metadata', download=download,
                             path_fn=get_card_groups_path,
                             build_fn=build_card_groups,
                             )

def get_card_metadata(cfg, set_code, download=True):
    return get_card_metadata_i(cfg, set_code, desc='metadata', download=download,
                             path_fn=get_card_metadata_path,
                             build_fn=build_card_raw_metadata_ext,
                             )

def get_card_picks(cfg, set_code, download=True):
    return get_card_metadata_i(cfg, set_code, desc='picks', download=download,
                             path_fn=get_card_picks_path,
                             build_fn=build_card_picks,
                             )

def download_set_code(cfg, set_code):
    download_and_extract_draft_dataset(cfg, set_code)    
    get_card_metadata(cfg,set_code)
    get_card_groups(cfg, set_code)        
    get_card_picks(cfg, set_code)


def download_all_set_codes(cfg):
    for set_code in DRAFT_TYPE_URLS:
        download_set_code(cfg, set_code)



class CardPickStats:
    """Streams a drafts CSV and counts, per card, offers and picks by scope.

    Repeated `process_drafts_file` calls accumulate, so a set split across
    several files sums correctly. They must share a header: combining files
    with different card pools into one table would produce counts whose
    denominators mean different things, so a mismatch raises rather than
    quietly taking the union.
    """

    # Column prefix marking "this card was in the pack at this pick".
    PACK_CARD_PREFIX = 'pack_card_'

    # 0-indexed pack number -> the label its first pick is reported under.
    FIRST_PICK_PACKS: Tuple[Tuple[int, str], ...] = (
        (0, 'p1p1'), (1, 'p2p1'), (2, 'p3p1'),
    )

    # Every scope counted, in output order.
    SCOPES: Tuple[str, ...] = ('p1p1', 'p2p1', 'p3p1', 'total')

    # The output contract, spelled out rather than derived.
    STAT_COLUMNS: Tuple[str, ...] = (
        'card_name',
        'p1p1_offered', 'p1p1_picked',
        'p2p1_offered', 'p2p1_picked',
        'p3p1_offered', 'p3p1_picked',
        'total_offered', 'total_picked',
    )

    # Columns needed beyond the pack_card_* block.
    REQUIRED_COLUMNS: Tuple[str, ...] = ('pack_number', 'pick_number', 'pick')

    # Rows per chunk. At ~300 pack_card columns read as float32 this is ~120 MB
    # per chunk, which is the trade being made -- larger chunks are marginally
    # faster and proportionally hungrier.
    DEFAULT_CHUNKSIZE = 100_000

    _UNKNOWN_PICKS = (
        '%d pick(s) name a card with no %s column, so they are absent from the'
        ' table: %s'
    )

    def __init__(
        self,
        chunksize: int = DEFAULT_CHUNKSIZE,
        progress: bool = True,
    ):
        if chunksize < 1:
            raise ValueError(f'chunksize must be positive, got {chunksize!r}')
        self.chunksize = chunksize
        self.progress = progress

        self.cards: List[str] = []
        self.sources: List[str] = []
        self.n_rows = 0
        # Card names picked with no pack_card_ column of their own -> count.
        self.unknown: Dict[str, int] = {}
        self.offered: Dict[str, np.ndarray] = {}
        self.picked: Dict[str, np.ndarray] = {}

        self._pack_columns: List[str] = []
        self._card_index: Optional[pd.Index] = None

    # ---- streaming ----
    def process_drafts_file(self, csv_path: str) -> None:
        """Accumulate counts from one drafts CSV.

        `offered` counts the *occasions* a card was in the pack, so a pack
        holding two copies counts once -- the question is how often the drafter
        had the choice, not how many copies existed. A cell reads as offered
        when it parses above zero, which keeps multi-copy packs from being
        dropped the way an `== 1` test would.

        `picked` counts rows whose `pick` is that card. Every row has exactly
        one pick, so `total_picked` sums to the row count over cards the header
        knows about.

        `progress` logs every chunk at INFO. Configure logging to see it.
        """
        self._begin(csv_path)
        dtypes: Dict[str, Any] = {c: np.float32 for c in self._pack_columns}
        dtypes['pack_number'] = np.int16
        dtypes['pick_number'] = np.int16

        reader = pd.read_csv(
            csv_path,
            usecols=list(self.REQUIRED_COLUMNS) + self._pack_columns,
            dtype=dtypes,
            chunksize=self.chunksize,
        )
        for chunk in reader:
            self._accumulate(chunk)
            self.n_rows += len(chunk)
            if self.progress:
                logging.info('card stats: %d rows', self.n_rows)

        self.sources.append(csv_path)
        if self.unknown:
            worst = sorted(self.unknown.items(), key=lambda kv: -kv[1])[:10]
            logging.warning(
                self._UNKNOWN_PICKS, sum(self.unknown.values()),
                self.PACK_CARD_PREFIX, worst,
            )

    def _begin(self, csv_path: str) -> None:
        """Read the header, then set up or re-validate the accumulators."""
        columns = self._header_columns(csv_path)
        if not self.cards:
            self._pack_columns = columns
            self.cards = [c[len(self.PACK_CARD_PREFIX):] for c in columns]
            self._card_index = pd.Index(self.cards)
            width = len(self.cards)
            self.offered = {s: np.zeros(width, dtype=np.int64)
                            for s in self.SCOPES}
            self.picked = {s: np.zeros(width, dtype=np.int64)
                           for s in self.SCOPES}
            return
        if columns != self._pack_columns:
            only_new = sorted(set(columns) - set(self._pack_columns))[:5]
            only_old = sorted(set(self._pack_columns) - set(columns))[:5]
            raise ValueError(
                f'{csv_path} has a different card pool from '
                f'{self.sources[0]}: new here {only_new}, missing here '
                f'{only_old}. Counts from different pools cannot share a table.'
            )

    def _header_columns(self, csv_path: str) -> List[str]:
        """The `pack_card_*` columns, read from the header alone."""
        header = pd.read_csv(csv_path, nrows=0)
        missing = [c for c in self.REQUIRED_COLUMNS
                   if c not in header.columns]
        if missing:
            raise KeyError(f'{csv_path} is missing column(s) {missing}')
        columns = [c for c in header.columns
                   if c.startswith(self.PACK_CARD_PREFIX)]
        if not columns:
            raise KeyError(
                f'{csv_path} has no {self.PACK_CARD_PREFIX}* columns, so no '
                'card can be said to have been offered. Is this a drafts CSV?'
            )
        return columns

    def _accumulate(self, chunk: pd.DataFrame) -> None:
        """Fold one chunk into every scope's counters."""
        # `> 0` rather than `== 1`: a pack with two copies still counts, and a
        # blank or NaN cell reads as absent rather than raising.
        present = chunk[self._pack_columns].to_numpy() > 0
        picks = chunk['pick']
        at_first = chunk['pick_number'].to_numpy() == 0
        packs = chunk['pack_number'].to_numpy()

        selections = [(np.ones(len(chunk), dtype=bool), 'total')]
        selections += [
            (at_first & (packs == pack), label)
            for pack, label in self.FIRST_PICK_PACKS
        ]
        for mask, scope in selections:
            if not mask.any():
                continue
            self.offered[scope] += present[mask].sum(axis=0)
            counts = picks[mask].value_counts()
            for name in counts.index.difference(self._card_index):
                self.unknown[name] = self.unknown.get(name, 0) + int(
                    counts[name])
            self.picked[scope] += counts.reindex(
                self.cards, fill_value=0).to_numpy()

    # ---- access ----
    def get_stats(self) -> List[Dict[str, Any]]:
        if not self.cards:
            raise ValueError(
                'no drafts processed yet; call process_drafts_file first'
            )
        counters = {'offered': self.offered, 'picked': self.picked}
        rows: List[Dict[str, Any]] = []
        for position, card in enumerate(self.cards):
            row: Dict[str, Any] = {'card_name': card}
            for column in self.STAT_COLUMNS[1:]:
                scope, kind = column.rsplit('_', 1)
                row[column] = int(counters[kind][scope][position])
            rows.append(row)
        rows.sort(key=lambda entry: entry['card_name'])
        logging.info(
            'card stats: %d cards over %d rows from %d file(s)',
            len(rows), self.n_rows, len(self.sources),
        )
        return rows

    def get_dataframe(self) -> pd.DataFrame:
        """`get_stats` as a frame: one row per card, `STAT_COLUMNS` in order."""
        return pd.DataFrame(self.get_stats())
