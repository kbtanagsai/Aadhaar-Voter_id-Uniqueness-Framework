import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import jellyfish
from tqdm import tqdm
import re
import os
import face_recognition

# =====================================================
# CONFIGURATION
# =====================================================
IMAGE_DIR = "voter_images"
FINAL_ROLL_FILE = "final_electoral_roll.csv"
DUPLICATE_FILE = "duplicate.csv"
FACE_TOLERANCE = 0.45 

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def normalize_indian_address(addr):
    if pd.isna(addr): return ""
    addr = str(addr).lower()
    replacements = {
        r"\bno\.?\b": "number", r"\bflr\b": "floor", r"\bapt\.?s?\b": "apartment",
        r"\bbld?g?\b": "building", r"\bqtrs\b": "quarters", r"\bph\b": "phase",
        r"\bst\.?\b": "street", r"\brd\.?\b": "road", r"\bave\.?\b": "avenue",
        r"\bln\.?\b": "lane", r"\bcr\.?\b": "cross", r"\bmkt\.?\b": "market",
        r"\bopp\.?\b": "opposite", r"\bnr\.?\b": "near", r"\bextn\.?\b": "extension",
        r"\bcol\.?\b": "colony", r"\bsect?\.?\b": "sector", r"\bgali\b": "lane",
        r"\bmarg\b": "road", r"\bbazaar\b": "bazar", r"\bchowk\b": "square",
        r"\bp\.?o\.?\b": "post office", r"\bp\.?o\.? box\b": "post office box",
        r"\bgpo\b": "general post office", r"\bpin\b": "postal index number",
        r"\bc/o\b": "care of", r",": " ", r"\.": " ", r"/": " ", r"-": " "
    }
    for k, v in replacements.items():
        addr = re.sub(k, v, addr)
    return " ".join(addr.split())

def get_face_encoding(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        return encodings[0] if len(encodings) > 0 else None
    except Exception:
        return None

# =====================================================
# MAIN WORKFLOW
# =====================================================
def run_integrated_verification():
    # 1. LOAD DATA
    print("--- Loading Aadhaar and Voter Rolls ---\n\n")
    try:
        aadhaar = pd.read_csv("aadhaar_roll.csv", dtype=str)
        voter = pd.read_csv("voter_roll.csv", dtype=str)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. PREPROCESSING
    if "aadhaar_unique_id" in aadhaar.columns:
        aadhaar = aadhaar.dropna(subset=["aadhaar_unique_id"]).drop_duplicates(subset=["aadhaar_unique_id"])

    voter["clean_address"] = voter["address"].apply(normalize_indian_address) if "address" in voter.columns else ""
    aadhaar["clean_address"] = aadhaar["address"].apply(normalize_indian_address) if "address" in aadhaar.columns else ""

    if "epic_number" in voter.columns:
        voter["epic_numeric"] = voter["epic_number"].str.extract(r'(\d+)').fillna(0).astype(int)
        voter = voter.sort_values("epic_numeric", ascending=False)

    final_records = []
    duplicate_records = []
    processed_epics = set()

    # 3. PHASE 1: AADHAAR & FUZZY DEDUPLICATION
    print("--- Phase 1: Details Verification ---\n")
    
    # Aadhaar Matching
    print("Processing Voters with Mapping...\n")
    voter_grouped = voter[voter["aadhaar_unique_id"].notna()].groupby("aadhaar_unique_id") if "aadhaar_unique_id" in voter.columns else {}
    aadhaar_lookup = aadhaar.set_index("aadhaar_unique_id").to_dict("index")

    for aid, adata in tqdm(aadhaar_lookup.items(), desc="Aadhaar Mapping"):
        if aid not in voter_grouped.groups: continue
        matches = voter_grouped.get_group(aid).copy()
        
        if len(matches) > 1:
            matches["name_sim"] = matches["name"].apply(lambda x: fuzz.token_sort_ratio(str(x).lower(), str(adata.get("name", "")).lower()))
            selected = matches.sort_values(["name_sim", "epic_numeric"], ascending=False).iloc[0].copy()
        else:
            selected = matches.iloc[0].copy()

        selected.update({"name": adata.get("name"), "age": adata.get("age"), "gender": adata.get("gender")})
        final_records.append(selected.to_dict())
        
        dupes = matches[matches["epic_number"] != selected["epic_number"]]
        if not dupes.empty: duplicate_records.extend(dupes.to_dict("records"))
        processed_epics.update(matches["epic_number"].tolist())

    # Fuzzy Logic 
    print("\nProcessing Remaining Voters with Fuzzy Logic...\n")
    remaining = voter[~voter["epic_number"].isin(processed_epics)].copy()
    if not remaining.empty:
        remaining["name_soundex"] = remaining["name"].apply(lambda x: jellyfish.soundex(str(x).split()[0]) if pd.notna(x) else "")
        for _, group in tqdm(remaining.groupby("name_soundex"), desc="Fuzzy Deduplication"):
            unique_in_group = []
            for _, row in group.iterrows():
                is_duplicate = False
                for u in unique_in_group:
                    n_score = fuzz.token_sort_ratio(str(row["name"]).lower(), str(u["name"]).lower())
                    a_score = fuzz.token_set_ratio(str(row["clean_address"]), str(u["clean_address"]))
                    if (n_score >= 90 and a_score >= 80):
                        is_duplicate = True
                        break
                if is_duplicate: duplicate_records.append(row.to_dict())
                else: unique_in_group.append(row.to_dict()); final_records.append(row.to_dict())

    # 4. PHASE 2: FACE-BASED RE-VERIFICATION
    print("\n--- Phase 2: Face Verification ---\n")
    print("Starting Face Matching...\n")
    temp_df = pd.DataFrame(final_records).fillna("")
    biometric_unique = []
    biometric_dupes = []
    known_encodings = []

    for _, row in tqdm(temp_df.iterrows(), total=len(temp_df), desc="Scanning Photos"):
        epic = row.get("epic_number", "").strip()
        img_path = os.path.join(IMAGE_DIR, f"{epic}.jpg")

        if epic == "" or not os.path.exists(img_path):
            biometric_unique.append(row.to_dict())
            continue

        encoding = get_face_encoding(img_path)
        if encoding is None:
            biometric_unique.append(row.to_dict())
            continue

        # Check against already accepted faces
        if len(known_encodings) > 0:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=FACE_TOLERANCE)
            if True in matches:
                biometric_dupes.append(row.to_dict())
                continue

        known_encodings.append(encoding)
        biometric_unique.append(row.to_dict())

    # 5. FINAL EXPORT
    final_df = pd.DataFrame(biometric_unique)
    all_duplicates = pd.concat([pd.DataFrame(duplicate_records), pd.DataFrame(biometric_dupes)], ignore_index=True)

    cols = ["name", "age", "gender", "epic_number", "constituency"]
    for df in [final_df, all_duplicates]:
        for c in cols: 
            if c not in df.columns: df[c] = ""

    final_df[cols].to_csv(FINAL_ROLL_FILE, index=False)
    all_duplicates[cols].drop_duplicates(subset=["epic_number"]).to_csv(DUPLICATE_FILE, index=False)

    print('\n=============================')
    print(f"\n✅ Verification Completed! \n Clean Electoral Records: {len(final_df)} \n Total Duplicate Records: {len(all_duplicates)}")
    print('\n=============================')
if __name__ == "__main__":
    run_integrated_verification()
    