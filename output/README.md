# Submitted files

Stored compressed because of GitHub's 100 MB file limit. To restore them byte-exact:

```bash
gunzip -c matching_results.tsv.gz > matching_results.tsv
cat candidate_pairs.tsv.gz.part* | gunzip > candidate_pairs.tsv
```
