# Local approved examples

This directory is intentionally shipped without vehicle photographs. Add an image here only when you are authorized to process it and its source, consent or other lawful basis, permitted uses, retention period, and deletion owner are recorded.

The application discovers local `.jpg`, `.jpeg`, and `.png` files in this directory as **Approved examples**. These files are ignored by Git and the Docker build context so that private test data is not committed or copied into an image accidentally. Provide production evaluation data through a separately controlled, access-restricted workflow; do not use this directory as a dataset store.

The photographs removed during the production-readiness refactor can remain reachable in existing Git history and remote caches. Before making the repository public, the repository owner must assess whether a coordinated history rewrite and cache cleanup are required. History rewriting is disruptive and is not performed automatically.
