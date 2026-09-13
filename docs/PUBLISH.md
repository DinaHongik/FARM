# Publishing the consolidated release

The prepared folder on the author's workstation is:

```text
/home/khbjane/Desktop/farm_rev1/FARM_release
```

Its `origin` must be `https://github.com/DinaHongik/FARM.git`. The parent workspace and the older `FARM/` research checkout are not the publishing folder.

The `final_update` commit removes manuscript sources, PDFs, and dedicated manuscript/rebuttal build scripts from the current release while preserving the code, examples, guides, and README benchmark values. Git authentication must use an account with access to DinaHongik/FARM; do not place a token in source files or in the remote URL.

Then publish from the prepared folder:

```bash
cd /home/khbjane/Desktop/farm_rev1/FARM_release
python3 scripts/publish.py
```

The helper verifies the destination and release, fetches the repository's existing default branch, and incorporates its history using a normal merge. Existing remote-only files are retained; overlapping files prefer this updated release. It then checks the integrated files and tests before pushing to the default branch. It never force-pushes. Authentication, unresolved merge conflicts, failed checks, or protected-branch restrictions stop the operation.

The release history was consolidated into one `final_update` commit after the repository's original first commit. The prepared workstation folder already uses this history. If you have another clone from before this cleanup, preserve any local edits separately and make a fresh clone before publishing; merging the old history would restore the removed commits.

For future edits:

```bash
git status
git add README.md docs src tests
git commit -m "Describe your FARM update"
python3 scripts/publish.py
```

Select the files you intend to commit. Do not stage `.env`, keys, private data, checkpoints, or generated output. The preparation configured a local pre-push hook that verifies both the repository destination and release files. After a new clone, enable the same hook with:

```bash
git config core.hooksPath .githooks
```

If the remote default branch requires pull requests, after the helper has incorporated its history and passed verification, push a review branch instead:

```bash
git push origin HEAD:refs/heads/release/final_update
```

Then open a pull request to the repository's default branch. A raw force push is not needed to replace the old README with this complete code release.
