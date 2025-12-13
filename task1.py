import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish


##################################################################################
#                                 Configuration                                  #
##################################################################################

DBLP_FILE = "DBLP1.csv"
SCHOLAR_FILE = "Scholar.csv"
MAPPING_FILE = "DBLP-Scholar_perfectMapping.csv"

TITLE_PREFIX_LEN_DEDUPLICATION = 6                  # chars used to block during deduplication
TITLE_PREFIX_LEN_BLOCKING = 4                       # length of title prefix for blocking
AUTHOR_MIN_OVERLAP = 1                              # min last-name overlap in author blocking
YEAR_TOLERANCE = 0                                  # year tolerance in year blocking
LEV_THRESHOLD = 0.80                                # compute Levenshtein only for pairs with cosine >= threshold



##################################################################################
#                                     Utils                                      #
##################################################################################

def clean_text(text):
    """Lowercase, remove punctuation and normalize whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)                            # remove punctuation
    text = re.sub(r"\s+", " ", text)                                # normalize whitespace
    return text.strip()

def tokenize(text):
    if not text:
        return []
    return clean_text(text).split()                                 # return a list of tokens split by space              

def extract_last_names(authors_str):
    """ Takes last token of each author as last name"""
    if pd.isna(authors_str):
        return []
    authors = re.split(r"[,;&]", str(authors_str))                  # split by common delimiters (comma, semicolon, ampersand)
    last_names = []
    for author in authors:
        author = author.strip()
        if not author:                                              # skip empty
            continue
        parts = author.split()
        last_names.append(clean_text(parts[-1]))                    # take last token as last name
    return last_names

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets"""
    if not set1 and not set2:
        return 1.0                                                  # both empty so they are identical
    if not set1 or not set2:
        return 0.0                                                  # one is empty, the other is not so they share nothing
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return inter / union
        
        
##################################################################################
#                                Data Preparation                                #
##################################################################################


def prepare_dataset(df, source_name):
    df = df.copy()
    # Normalize title
    df['title_clean'] = df['title'].apply(clean_text)
    df['title_tokens'] = df['title_clean'].apply(tokenize)                      # create a new column with title tokens
    
    # Last names
    df['author_lastnames'] = df['authors'].apply(extract_last_names)

    # Year parsing
    df['year_parsed'] = pd.to_numeric(df.get('year', None), errors='coerce')    # convert to numeric (invalid parsing will be set as NaN)

    # Venue normalized
    df['venue_clean'] = df.get('venue', '').apply(clean_text)

    df['source'] = source_name                                                  # add source column so we know if it comes from DBLP or Scholar                                     
    return df


##################################################################################
#                                 Deduplication                                  #
##################################################################################

# We deduplicate the dataset by grouping records that are similar
# We create a "block key" to limit comparisons
# Then we uses Jaccard similarity on title tokens within each block to identify duplicates
# It returns the DataFrame with only unique records

def deduplicate_dataset_blocking(df, block_chars=TITLE_PREFIX_LEN_DEDUPLICATION, sim_threshold=0.95):
    """Deduplicate dataset using blocking keys and Jaccard inside blocks"""
    df = df.copy().reset_index(drop=True)
    n_original = len(df)

    # Build block key: first block_chars of title + year + first author lastname (if present)
    def make_block_key(row):
        title = row['title_clean'] or ''
        key_parts = [title[:block_chars]]
        year = row['year_parsed']
        key_parts.append(str(int(year)) if not pd.isna(year) else 'na')
        first_author = row['author_lastnames'][0] if row['author_lastnames'] else 'na'
        key_parts.append(first_author[:10])
        return '_'.join(key_parts)

    df['dedup_block'] = df.apply(make_block_key, axis=1)

    groups = []
    block_index = defaultdict(list)
    for idx, key in enumerate(df['dedup_block']):
        block_index[key].append(idx)                                # build the dictionary of block_key -> list of row indices

    for key, indices in block_index.items():
        if len(indices) == 1:                                       # only one record in block
            groups.append([indices[0]])
            continue

        # Compare only within block
        local_used = set()
        for i in indices:
            if i in local_used:
                continue
            group = [i]
            tokens_i = set(df.at[i, 'title_tokens'])
            for j in indices:
                if j <= i or j in local_used:
                    continue
                tokens_j = set(df.at[j, 'title_tokens'])
                sim = jaccard_similarity(tokens_i, tokens_j)
                if sim >= sim_threshold:
                    group.append(j)
                    local_used.add(j)
            local_used.add(i)
            groups.append(group)

    unique_indices = [g[0] for g in groups]
    df_unique = df.loc[unique_indices].reset_index(drop=True)
    return df_unique, len(groups), n_original


##################################################################################
#                                 Blocking Rules                                 #
##################################################################################

def blocking_title_prefix(df_dblp, df_scholar, prefix_len=TITLE_PREFIX_LEN_BLOCKING):
    index2 = defaultdict(list)
    for i, row in enumerate(df_scholar.itertuples(index=False)):
        title = row.title_clean
        if title and len(title) >= prefix_len:
            index2[title[:prefix_len]].append(row.id)

    pairs = set()
    for i, row in enumerate(df_dblp.itertuples(index=False)):
        title = row.title_clean
        if title and len(title) >= prefix_len:
            for id2 in index2.get(title[:prefix_len], []):          # loop on all ids from scholar with same title prefix
                pairs.add((row.id, id2))                            # add pair

    return pairs

# Exemple : df_scholar ids & auteurs
# 1:["Smith","Doe"], 2:["Doe","Lee"], 3:["Brown","Smith"]
# index2 -> {"Smith":[1,3], "Doe":[1,2], "Lee":[2], "Brown":[3]}
def blocking_author_overlap(df_dblp, df_scholar, min_overlap=AUTHOR_MIN_OVERLAP):
    index2 = defaultdict(list)
    scholar_rows = {row.id: row for row in df_scholar.itertuples(index=False)}     # create a dictionary id -> row for scholar

    for i, row in enumerate(df_scholar.itertuples(index=False)):
        for author in set(row.author_lastnames):
            if author:
                index2[author].append(row.id)

    pairs = set()

    for i, row1 in enumerate(df_dblp.itertuples(index=False)):
        matched_ids = set()
        ln1 = set(row1.author_lastnames)

        # Gather candidates
        for author in ln1:
            matched_ids.update(index2.get(author, []))

        # Validate overlap
        for id2 in matched_ids:
            row2 = scholar_rows[id2]
            ov = len(ln1.intersection(row2.author_lastnames))
            if ov >= min_overlap:
                pairs.add((row1.id, id2))

    return pairs

def blocking_year(df_dblp, df_scholar, year_tolerance=YEAR_TOLERANCE):
    index2 = defaultdict(list)

    for i, row in enumerate(df_scholar.itertuples(index=False)):
        year = row.year_parsed
        if not pd.isna(year):
            index2[int(year)].append(row.id)

    pairs = set()

    for i, row1 in enumerate(df_dblp.itertuples(index=False)):
        year = row1.year_parsed
        if not pd.isna(year):
            for dy in range(-year_tolerance, year_tolerance + 1):
                for id2 in index2.get(int(year) + dy, []):
                    pairs.add((row1.id, id2))

    return pairs

def compute_similarities_vectorized(candidate_pairs, dblp_df, scholar_df):
    """Compute similarities for candidate pairs and returns the DataFrame with scores"""
    # Build corpus and fit TF-IDF once
    titles_dblp = list(dblp_df['title_clean'])
    titles_scholar = list(scholar_df['title_clean'])
    ids_dblp = list(dblp_df['id'])
    ids_scholar = list(scholar_df['id'])

    corpus = titles_dblp + titles_scholar
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)

    n_db = len(titles_dblp)

    # Build maps: id -> row-index-in-tfidf
    map_dblp = {idv: i for i, idv in enumerate(ids_dblp)}
    map_scholar = {idv: n_db + i for i, idv in enumerate(ids_scholar)}

    # Prepare arrays of indices for candidate pairs, filtering missing ids
    idxs_db = []
    idxs_sch = []
    filtered_pairs = []
    for i, (id_db, id_sch) in enumerate(candidate_pairs):
        if id_db in map_dblp and id_sch in map_scholar:
            idxs_db.append(map_dblp[id_db])
            idxs_sch.append(map_scholar[id_sch])
            filtered_pairs.append((id_db, id_sch))

    if not filtered_pairs:
        return pd.DataFrame([])

    # Extract TF-IDF rows (order preserved)
    tfidf_db = tfidf[idxs_db]
    tfidf_sch = tfidf[idxs_sch]

    # Compute cosine similarity components
    numer = tfidf_db.multiply(tfidf_sch).sum(axis=1).A1

    # Precompute norms for denominator
    sq = tfidf.power(2).sum(axis=1).A1
    norms = np.sqrt(sq)
    norms_db = norms[np.array(idxs_db)]
    norms_sch = norms[np.array(idxs_sch)]
    denom = norms_db * norms_sch

    # Compute cosine similarity
    cosine = np.zeros_like(numer, dtype=float)
    mask = denom > 0
    cosine[mask] = numer[mask] / denom[mask]

    # Prepare other measures (Jaccard on tokens, Dice, author_jaccard, year match, venue_jaccard)
    records = []
    dblp_map_rows = {r['id']: r for r in dblp_df.to_dict('records')}
    scholar_map_rows = {r['id']: r for r in scholar_df.to_dict('records')}

    total_pairs = len(filtered_pairs)
    for i, (id_db, id_sch) in enumerate(filtered_pairs):
        rec_db = dblp_map_rows[id_db]
        rec_sch = scholar_map_rows[id_sch]

        tokens_db = set(rec_db['title_tokens'])
        tokens_sch = set(rec_sch['title_tokens'])
        jacc = jaccard_similarity(tokens_db, tokens_sch)
        inter = len(tokens_db.intersection(tokens_sch))
        dice = 2 * inter / (len(tokens_db) + len(tokens_sch)) if (len(tokens_db) + len(tokens_sch)) > 0 else 0.0

        auth_db = set(rec_db['author_lastnames'])
        auth_sch = set(rec_sch['author_lastnames'])
        auth_jacc = jaccard_similarity(auth_db, auth_sch)

        venue_db = set(str(rec_db.get('venue_clean','')).split())
        venue_sch = set(str(rec_sch.get('venue_clean','')).split())
        venue_jacc = jaccard_similarity(venue_db, venue_sch)  # small improvement

        ydb = rec_db['year_parsed']
        ysch = rec_sch['year_parsed']
        year_match = 1.0 if (not pd.isna(ydb) and not pd.isna(ysch) and int(ydb) == int(ysch)) else 0.0

        records.append({
            'id_dblp': id_db,
            'id_scholar': id_sch,
            'cosine': float(cosine[i]),
            'jaccard': jacc,
            'dice': dice,
            'author_jaccard': auth_jacc,
            'venue_jaccard': venue_jacc,
            'year_match': year_match
        })

    sim_df = pd.DataFrame(records)

    # Compute Levenshtein for high-cosine pairs only
    consider = sim_df[sim_df['cosine'] >= LEV_THRESHOLD]
    if not consider.empty:
        levs = []
        for i, (_, row) in enumerate(consider.iterrows()):
            t1 = dblp_map_rows[row['id_dblp']]['title_clean']
            t2 = scholar_map_rows[row['id_scholar']]['title_clean']
            dist = jellyfish.levenshtein_distance(t1, t2)
            max_len = max(len(t1), len(t2))
            lev_sim = 1 - (dist / max_len) if max_len > 0 else 0
            levs.append(lev_sim)
        sim_df.loc[consider.index, 'levenshtein'] = levs
    else:
        sim_df['levenshtein'] = 0.0

    return sim_df


##################################################################################
#                             Matching & Evaluation                              #
##################################################################################

def evaluate_matching(sim_df, ground_truth_set, thresholds=[0.7, 0.8, 0.9, 0.95], score_col='cosine'):
    results = []
    for thr in thresholds:
        preds = set(tuple(x) for x in sim_df[sim_df[score_col] >= thr][['id_dblp', 'id_scholar']].values)
        tp = len(preds & ground_truth_set)
        fp = len(preds - ground_truth_set)
        fn = len(ground_truth_set - preds)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results.append({
            'threshold': thr,
            'predicted': len(preds),
            'true_positives': tp,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
    return pd.DataFrame(results)


##################################################################################
#                                      Main                                      #
##################################################################################

def main():
    
    #---------------- Load & Prepare Data ----------------
    dblp = pd.read_csv(DBLP_FILE, encoding='latin-1')
    scholar = pd.read_csv(SCHOLAR_FILE, encoding='utf-8')

    print(f"Initial records: DBLP={len(dblp)}, Scholar={len(scholar)}")

    dblp_prep = prepare_dataset(dblp, 'DBLP')
    scholar_prep = prepare_dataset(scholar, 'Scholar')

    #------------- ID Assignment for Deduplication -------------
    if 'id' not in dblp_prep.columns:
        dblp_prep = dblp_prep.reset_index().rename(columns={'index': 'id'})
        dblp_prep['id'] = dblp_prep['id'].apply(lambda x: f"DBLP_{x}")
    if 'id' not in scholar_prep.columns:
        scholar_prep = scholar_prep.reset_index().rename(columns={'index': 'id'})
        scholar_prep['id'] = scholar_prep['id'].apply(lambda x: f"SCH_{x}")

    # ---------------- Deduplication ----------------
    dblp_unique, dblp_groups, dblp_orig = deduplicate_dataset_blocking(dblp_prep)
    print(f"DBLP: {dblp_orig} -> {len(dblp_unique)} unique ({dblp_groups} groups)")

    scholar_unique, sch_groups, sch_orig = deduplicate_dataset_blocking(scholar_prep)
    print(f"Scholar: {sch_orig} -> {len(scholar_unique)} unique ({sch_groups} groups)")

    # ---------------- Blocking ----------------
    pairs_title = blocking_title_prefix(dblp_unique, scholar_unique)
    pairs_author = blocking_author_overlap(dblp_unique, scholar_unique)
    pairs_year = blocking_year(dblp_unique, scholar_unique)

    print(f"Pairs - title: {len(pairs_title)}, author: {len(pairs_author)}, year: {len(pairs_year)}")

    all_pairs = set().union(pairs_title, pairs_author, pairs_year)
    print(f"Union candidate pairs: {len(all_pairs)}")

    # ---------------- Load Ground Truth ----------------
    gt = pd.read_csv(MAPPING_FILE)
    gt_set = set(zip(gt['idDBLP'], gt['idScholar']))


    if len(gt_set) > 0:
        retained = len(all_pairs & gt_set)
        recall_blocking = retained / len(gt_set)
        print(f"Gold matches retained by blocking: {retained} / {len(gt_set)} ({recall_blocking:.2%})")

    # ---------------- Similarity Scoring ----------------
    candidate_list = list(all_pairs)

    sim_df = compute_similarities_vectorized(candidate_list, dblp_unique, scholar_unique)
    if sim_df.empty:
        print("No candidate similarities computed. Exiting.")
        return

    print("\n=== SIMILARITY SCORING REPORT ===")
    print(f"Total candidate pairs scored: {len(sim_df)}")
    print(f"\nPairs with cosine ≥ 0.95: {len(sim_df[sim_df['cosine'] >= 0.95])}")
    print(f"Pairs with cosine ≥ 0.90: {len(sim_df[sim_df['cosine'] >= 0.90])}")
    print(f"Pairs with cosine ≥ 0.80: {len(sim_df[sim_df['cosine'] >= 0.80])}")
    print(f"Pairs with cosine ≥ 0.70: {len(sim_df[sim_df['cosine'] >= 0.70])}")

    print("\n--- Similarity Measures Summary ---")
    for col in ['cosine', 'jaccard', 'dice', 'levenshtein', 'author_jaccard', 'venue_jaccard', 'year_match']:
        if col in sim_df.columns:
            print(f"\n{col.upper()}:")
            print(sim_df[col].describe())

    # ---------------- Matching & Evaluation ----------------

    # Evaluation with Cosine
    print("\n=== COSINE SIMILARITY ===")
    eval_cosine = evaluate_matching(sim_df, gt_set, thresholds=[0.70, 0.80, 0.90, 0.95], score_col='cosine')
    print(eval_cosine)

    # Evaluation with Jaccard
    print("\n=== JACCARD SIMILARITY ===")
    eval_jaccard = evaluate_matching(sim_df, gt_set, thresholds=[0.70, 0.80, 0.90, 0.95], score_col='jaccard')
    print(eval_jaccard)

    # Evaluation with Dice
    print("\n=== DICE SIMILARITY ===")
    eval_dice = evaluate_matching(sim_df, gt_set, thresholds=[0.70, 0.80, 0.90, 0.95], score_col='dice')
    print(eval_dice)

    # Evaluation with Levenshtein
    print("\n=== LEVENSHTEIN SIMILARITY ===")
    eval_lev = evaluate_matching(sim_df, gt_set, thresholds=[0.70, 0.80, 0.90, 0.95], score_col='levenshtein')
    print(eval_lev)

    # Evaluation with Author Jaccard
    print("\n=== AUTHOR JACCARD SIMILARITY ===")
    eval_author = evaluate_matching(sim_df, gt_set, thresholds=[0.70, 0.80, 0.90, 0.95], score_col='author_jaccard')
    print(eval_author)

    # ---------------- Save Results ----------------
    sim_df.to_csv('scores/similarity_scores.csv', index=False)
    eval_cosine.to_csv('scores/evaluation_cosine.csv', index=False)
    eval_jaccard.to_csv('scores/evaluation_jaccard.csv', index=False)
    eval_dice.to_csv('scores/evaluation_dice.csv', index=False)
    eval_lev.to_csv('scores/evaluation_levenshtein.csv', index=False)
    eval_author.to_csv('scores/evaluation_author.csv', index=False)

    print("\nResults saved:")

    # Comparison summary
    print("\n=== BEST F1 SCORES COMPARISON ===")
    comparison = pd.DataFrame({
        'Measure': ['Cosine', 'Jaccard', 'Dice', 'Levenshtein', 'Author'],
        'Best_F1': [
            eval_cosine['f1'].max(),
            eval_jaccard['f1'].max(),
            eval_dice['f1'].max(),
            eval_lev['f1'].max(),
            eval_author['f1'].max()
        ],
        'Best_Threshold': [
            eval_cosine.loc[eval_cosine['f1'].idxmax(), 'threshold'],
            eval_jaccard.loc[eval_jaccard['f1'].idxmax(), 'threshold'],
            eval_dice.loc[eval_dice['f1'].idxmax(), 'threshold'],
            eval_lev.loc[eval_lev['f1'].idxmax(), 'threshold'],
            eval_author.loc[eval_author['f1'].idxmax(), 'threshold']
        ]
    })
    print(comparison)
    comparison.to_csv('scores/comparison_summary.csv', index=False)

if __name__ == '__main__':
    main()
