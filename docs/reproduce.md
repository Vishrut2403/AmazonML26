# Reproducing the submission

Every command runs from the repository root. Steps that work through many chunks skip the chunks that are already
done, so you can restart them. On a 13 GB machine, run one heavy step at a time.

## Setup

You need Python 3.12 on Linux. The LightGBM steps run on CPU (16 threads and 13 GB of RAM were enough). The
cross-encoder steps need an NVIDIA GPU (6 GB was enough).

```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
```

The challenge data isn't included. Put the challenge kit at `student_resource/`, so the data is in
`student_resource/dataset/{train,test}/` and the validator is in `student_resource/utils/`. Intermediate files go to
`work/`, and the submission files go to `output/`.

## 1. Clean the text and find candidates

```
python src/prep.py train && python src/prep.py test          # -> work/{train,test}.parquet
python src/block.py train 10 && python src/block.py test 10  # top 10 S1 per S2/S3 record -> work/{train,test}_cand/
```

## 2. LightGBM, two passes

```
python src/match.py train                  # -> work/model.txt
python src/stage2.py train                 # -> work/stage2/ models, work/stage2/val_scored.parquet
python src/stage2.py predict               # -> work/stage2/test_scored/ (score of every test pair)
```

We made the submitted stage-2 scores before `match.py` got its last three house-number features (`num_diff`,
`num_up`, `o_no_num`). To match the submission more closely, remove those three from `FEATS` in `src/match.py`.

## 3. Cross-encoders

```
python src/ce_data.py                      # training pairs -> work/ce/{train,val}.parquet
python src/ce_train.py                     # multilingual-e5-small -> work/ce/model
python src/ce_predict.py tune && python src/ce_predict.py predict      # -> work/ce/test_scored/
CE_DIR=ce_base CE_BASE=intfloat/multilingual-e5-base CE_BATCH=64 CE_LR=3e-5 CE_FREEZE_EMB=1 python src/ce_train.py --no-val
CE_DIR=ce_base python src/ce_rescore.py val && CE_DIR=ce_base python src/ce_rescore.py test
CE_DIR=ce_base2 CE_BASE=work/ce_base/model CE_BATCH=64 CE_LR=2e-5 CE_FREEZE_EMB=1 python src/ce_train.py --no-val
CE_DIR=ce_base2 python src/ce_rescore.py val && CE_DIR=ce_base2 python src/ce_rescore.py test
```

The base model only rescores the records the small model is unsure about (about 8%). `ce_base2` is a second epoch
on the same pairs.

## 4. Extra candidates from embedding search and Indian-script names

```
python src/dense.py search train && python src/dense.py search test
python src/dense.py score train && python src/dense.py score test
python src/translit.py learn               # word dictionary -> work/translit.json
CE_DIR=ce_base2 python src/translit.py run train && CE_DIR=ce_base2 python src/translit.py run test
CE_DIR=ce_base2 python src/dense.py final translit                     # merged test scores -> work/merged/translit/
```

## 5. Final model (stack)

```
CE_DIR=ce_base2 python src/tools/val_merge.py work/val_merged_ce.parquet
python src/stack.py val work/val_merged_ce.parquet work/stage2/val_scored.parquet
python src/stack.py test work/merged/translit "work/stage2/test_scored/*.parquet" "work/stage2/test_scored/*.parquet" 0.723/0.3
                                           # -> output/matching_results_stack_fr0.723_0.3.tsv
```

India/US records are decided by the stack. France records are decided by the stage-2 scores at 0.723/0.3 for now;
step 6 replaces them.

## 6. France

```
python src/tools/france_pairs.py           # likely French pairs -> work/france_pairs.parquet
python src/tools/france_ce.py bundle work/fr_ce_input.parquet          # French rows of the step 3 test scores
python src/tools/france_ce.py data work/fr_ce_input.parquet            # -> work/ce_fr/train.parquet
CE_DIR=ce_fr CE_BASE=work/ce/model CE_BATCH=64 CE_LR=2e-5 python src/ce_train.py --no-val
python src/tools/france_score.py score "work/stage2/test_scored/*.parquet"   # top 3 per French record -> work/fr_ce_scores.parquet
python src/tools/france_score.py variants output/matching_results_stack_fr0.723_0.3.tsv work/submissions/C2v85 0.85 0.5
                                           # France: stage 2 at 0.85/0.5, kept if the France cross-encoder gives >= 0.5
python src/tools/initials.py test          # initials and website names -> work/initials_pairs_test.parquet
python src/tools/assemble.py work/submissions/C2v85_agree/upload/matching_results.tsv work/submissions/K1.tsv \
    override:work/initials_pairs_test.parquet
python src/tools/fr_org.py work/submissions/K1.tsv work/submissions/final.tsv   # French organisation-word rules
python src/tools/candidates.py work/submissions/final.tsv              # -> output/matching_results.tsv, output/candidate_pairs.tsv
```

## 7. Check the files with the challenge validator

```
cd student_resource
python utils/validate_submission.py --matching ../output/matching_results.tsv --test-dir dataset/test --check-ids
python utils/validate_submission.py --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

The second check keeps every candidate id in memory, which took about 9 GB for our 63M candidate pairs.

## How close a rerun gets

During the challenge, some steps ran on a second machine, and the models were trained once with random sampling. A
fresh run therefore gives a file close to our submission, but not identical to it. We checked step 1 from a clean
clone: it gives the same text for every India and US record as our submitted run.

## What each file does

| File | What it does |
|---|---|
| `src/prep.py` | Turns every script into Latin letters, lowercases, expands street abbreviations, removes legal words to get a `core` name, pulls out house numbers |
| `src/block.py` | Each S2/S3 record takes its 8 rarest tokens and keeps the 10 S1 records in its country that share the most (by summed IDF) |
| `src/match.py`, `src/stage2.py` | LightGBM on name, address and number similarities; the second pass adds how the record ranks against the other records that want the same S1 business |
| `src/ce_*.py` | Fine-tune multilingual-e5 as a cross-encoder on (record, candidate) text |
| `src/dense.py` | Nearest S1 records by e5 embeddings, for matches blocking missed |
| `src/translit.py` | Maps Indian-script names back to English words with a dictionary learned from training pairs, then blocks and scores them again |
| `src/stack.py` | Final LightGBM over all scores; a record goes to its best candidate when the score is at least 0.3 and at least 0.5 ahead of the second-best |
| `src/tools/france_pairs.py`, `france_ce.py`, `france_score.py` | France training data, the France cross-encoder scores, and the agreement rule |
| `src/tools/fr_org.py`, `src/tools/edits.py` | French organisation-word and decoy rules |
| `src/tools/initials.py` | Matches records whose name is only initials or a website name |
| `src/tools/assemble.py`, `candidates.py`, `val_merge.py` | Combine files and write the candidate file |
