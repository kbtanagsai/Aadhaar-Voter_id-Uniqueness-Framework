# Aadhaar-Voter_id-Uniqueness-Framework

AI-powered voter deduplication system that ensures one voter corresponds to one identity by combining Aadhaar verification, fuzzy matching, phonetic analysis, address normalization, and facial recognition.

## Features

- Aadhaar-based voter verification
- Fuzzy duplicate detection
- Address normalization
- Soundex phonetic matching
- Face recognition validation
- Clean electoral roll generation
- Duplicate record reporting

## Tech Stack

Python, Pandas, NumPy, RapidFuzz, Jellyfish, Face Recognition

## Installation

```bash
pip install pandas numpy rapidfuzz jellyfish tqdm face_recognition opencv-python
```

## Run

```bash
python detect.py
```

## Output

- `final_electoral_roll.csv` – Verified voter records
- `duplicate.csv` – Detected duplicate records

## Workflow

Aadhaar Verification → Address Normalization → Fuzzy Matching → Soundex Matching → Face Verification → Final Electoral Roll
