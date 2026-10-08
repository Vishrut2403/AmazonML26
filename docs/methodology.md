# Methodology

This is how the pipeline works, why we built it this way, and what we measured along the way. To run it, see
[reproduce.md](reproduce.md).

## The task

Business records come from three sources with no shared ids. Source 1 (S1) is a clean reference list. For every S1
business, we had to find the Source 2 and Source 3 (S2/S3) records that describe the same business. The score is
F0.5 per S1 business, averaged over all of them, so a wrong match costs more than a missed one.

## What the data showed

The training set has 2.2M S1 businesses, 10.3M S2/S3 records and 7.64M true pairs. A few things stood out early:

- Each S2/S3 record belongs to at most one S1 business, and always one in the same country. So we treated the task
  as "for each S2/S3 record, pick one S1 business or none".
- 26% of S2/S3 records match nothing in train, and test has about twice as many of these per S1 business.
- True copies get typos, lose accents, have legal forms added, dropped or swapped (Pvt Ltd / Private Limited), get
  words reordered or filler words added (Services, Sri, Mr), and sometimes show up as initials (`VJ`), as a website
  (`vieillesjeunes.com`), as an invented brand name at the real address, or in an Indian script. Addresses get
  shortened, reordered or dropped (3% are empty).
- Decoys, the records that match nothing, are almost always a copy of a real business with the house number moved
  up by 1 to 60 and one name part changed: a different legal form, or a filler from a separate list (Holdings,
  Group, Enterprises). The two filler lists don't overlap. Added at the same address, Services, Sri and Mr were true
  matches 100% of the time, and Holdings, Group and Enterprises 0% of the time.
- France only appears in test (15% of S1), so it has no labels. French names follow "core words + organisation
  word + legal form" (`lille amicale sarl`), and French decoys also swap the organisation word (`lille club sarl`).

## Pipeline

| Step | What it does | Files |
|---|---|---|
| Clean text | Every script to Latin letters, lowercase, expand abbreviations, strip legal words to get a "core" name | `prep.py` |
| Find candidates | Rare-word blocking, embedding search, an Indian-script dictionary, initials and website names | `block.py`, `dense.py`, `translit.py`, `tools/initials.py` |
| Score pairs | LightGBM in two passes, and two fine-tuned multilingual-e5 cross-encoders | `match.py`, `stage2.py`, `ce_*.py` |
| Combine | A final LightGBM over all scores | `stack.py` |
| Decide | Each record goes to its best candidate when that candidate is clearly ahead | `stack.py` |
| France | Its own cross-encoder, an agreement filter, and rules about French decoys | `tools/france_*.py`, `tools/fr_org.py` |

## Finding candidates

| Method | How it works | Result |
|---|---|---|
| Rare-word blocking | Each S2/S3 record takes its 8 rarest tokens (name and address words, number + street, sound keys so `imdiyn emtrpraiss` meets `indian enterprises`) and keeps the 10 S1 records in its country with the highest summed IDF | 96.5% of true pairs in the top 10 |
| Embedding search | Nearest S1 records by multilingual-e5-small embeddings (not fine-tuned) | 98.48% of true pairs on validation, together with blocking |
| Indian-script dictionary | A word dictionary learned from the training pairs maps transliterated words back (`piraivet limitet` → `private limited`), and those records are blocked again | Records whose name matches the true S1 name went from 6.5% to 94.2% |
| Initials and website names | Takes the S1 record at the same address whose initials or squashed name fit, if exactly one fits | Precision 1.000 on training data |

That gives 63.2M candidate pairs on test, and every final match is one of them.

## Scoring

The first LightGBM pass (`match.py`) uses about 40 features computed from the two records' text. For names, these
are edit, token-set, token-sort, partial and Jaro-Winkler similarity on the core name, the squashed name and the
sound keys. Addresses get the same similarities, with and without numbers. There are also house-number checks
(first number equal, overlap, containment), the blocking score and rank, and how many records want the same S1
business. The last group compares the record with the strongest other record for that business.

The second pass (`stage2.py`) adds the first-pass scores of the records competing for the same S1 business. A record
up against a very confident competitor is less likely to be the match.

The cross-encoders (`ce_*.py`) read "record text [SEP] candidate text" and output a match score.
`multilingual-e5-small` (118M parameters) scores every record, and `multilingual-e5-base` (278M) rescores the 8%
the small one is unsure about. Both are MIT licensed. The 6 GB GPU was the real limit here.

The final model (`stack.py`) is a LightGBM over the cross-encoder score, the stage-2 score, a few raw text checks
(exact name and address, legal forms, house number) and how each pair ranks among the record's candidates. It is
trained on 5-fold out-of-fold predictions on validation.

The decision rule gives a record its best candidate when the final score is at least 0.3 and at least 0.5 ahead of
the second-best. Otherwise the record stays unmatched. We tuned both numbers for the metric on a validation set: 1%
of the training S1 businesses (22,224), matched against the full training pool.

## France

With no French labels, we couldn't train or validate on France directly, so it got three extra steps.

First, a France cross-encoder. We fine-tuned the small cross-encoder again on French test pairs with stand-in
labels. The positives were pairs that rules find (same name and house number, or same address and similar names),
plus records the base cross-encoder was almost sure about (best score at least 0.99, runner-up at most 0.01). The
other top candidates of those records became negatives. We also mixed in 250k India/US training pairs so it didn't
forget what it already knew.

Second, an agreement filter. A French match is kept only when stage 2 accepts it (0.85, margin 0.5) and the France
cross-encoder gives it at least 0.5. This mostly removed near-copy decoys and moved the leaderboard from 0.978 to
0.982.

Third, rules about the organisation word (`fr_org.py`). They remove matches where the organisation word was swapped
(`amicale` to `club`) or added, but keep the ones where it was replaced by a filler (`club` to `& fils`), because
those are true. Records that are still unmatched get their only S1 business at the same address, provided the names
differ only by typos, the record is a one-word brand name, or the France cross-encoder is sure (0.9 or more). The
last rule removes matches where the house number went up by 1 to 60 and the record adds a decoy filler or a legal
form.

We checked every France rule on the public leaderboard before keeping it.

## Results

| Version | Change | Validation F0.5 | Public leaderboard |
|---|---|---|---|
| v1 | Word blocking (top 20) + LightGBM, 23 features | 0.8816 | |
| v3 | Better blocking (sound keys, word pairs), number fixes, partner features | 0.9546 | 0.940 |
| v4 | Stronger LightGBM | 0.9603 | 0.948 |
| CE | Second LightGBM pass and cross-encoders | | 0.963 |
| France stage 2 | France decided by stage 2, French address words cleaned | | 0.972 → 0.978 |
| Agreement | French matches kept only if the France cross-encoder agrees | | 0.982 |
| Stack | Embedding search, Indian-script dictionary, final LightGBM | 0.9894 | |
| Initials | Initials and website-name matcher | | 0.982661 |
| Org swaps | Remove French organisation-word swaps and additions | | 0.985243 |
| M4 | Larger learned vocabulary, same-address recoveries | | 0.986914 |
| Final | Remove French matches with a raised house number and an added decoy filler or legal form | | 0.986971 |

Most of the remaining validation loss comes from missed matches. These are mostly records with an empty address and
a name that many S1 businesses share, plus invented brand names that blocking never found. Wrong matches are rarer.
They are mostly decoys with a changed house number and an added legal form or filler (`studio 79 yoga` at 615 vs
`studio 79 yoga ltd` at 61): 187 of the 75,340 accepted pairs. Every rule we tried against them removed more true
matches than wrong ones.

These things didn't help:

- tuning the stack or changing random seeds (everything stayed within 0.0001)
- training the cross-encoder on more pairs (+0.001 after many hours)
- giving an S1 business with no match its best leftover record
- retuning the threshold for the test's extra decoys
- translating French words to English before scoring
- recovering French records from the embedding search or from any record at the same address (0.982 → 0.981)
- removing French matches where the organisation word became a filler (0.985 → 0.984)

## Setup and rules

Everything ran on a laptop with 16 CPU threads, 13 GB of RAM and a 6 GB GPU, plus a MacBook for some France and
stage-2 runs. The models are `intfloat/multilingual-e5-small`, `intfloat/multilingual-e5-base` (both MIT) and
LightGBM, all well under the 8B-parameter limit. We only used the provided training and test files, with no external
data, APIs or lookups.
