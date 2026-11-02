# Task Completion Verification

Before considering any audit or feature task complete:
1. Run fusion unit tests: `python -m unittest tests/test_fusion.py`
2. Run FER evaluation script: `python scripts/evaluate_fer.py`
3. Validate Web app import and sanity check: `python -c "import app; print('App syntax OK')"`
4. Verify Git status is clean of unintended binaries, keys, and pycache: `git status`
