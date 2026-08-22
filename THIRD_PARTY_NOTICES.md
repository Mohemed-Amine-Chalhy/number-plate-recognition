# Third-party notices and release review

This repository depends on third-party software and may use third-party or derived model artifacts. Those components retain their own copyright and license terms. The repository's MIT license applies only to repository-authored material where stated; it does not relicense dependencies, base models, custom weights, or datasets.

This file is a review checklist and notice summary, not legal advice. Generate and review a dependency inventory/SBOM from the final locked environment and container for every release; package names below are not a complete legal attribution report.

## Ultralytics and model licensing

The application uses the Ultralytics Python package and YOLO model artifacts. Ultralytics publishes AGPL-3.0 and Enterprise licensing options and states terms for its software and trained models. Review the current [Ultralytics licensing information](https://www.ultralytics.com/license), [AGPL-3.0 terms](https://www.ultralytics.com/legal/agpl-3-0-software-license), [Enterprise terms](https://www.ultralytics.com/legal/enterprise-software-license), the package/source license for the exact pinned version, and the provenance of each weight.

Before production or commercial use, document one approved path:

- comply with all applicable AGPL-3.0 obligations for the complete deployed/distributed work;
- obtain sufficient Ultralytics Enterprise/commercial rights; or
- replace the Ultralytics dependency and/or model artifacts with an approved alternative.

The project maintainers cannot infer that decision from the repository's MIT license. Obtain qualified advice for the planned use and distribution model.

## Custom and bundled weights

The repository currently contains `yolov10n.pt`, `license_plate_detector.pt`, and `PlateReaderyolo.pt`. Their presence and verified hashes are not evidence of redistribution or production-use permission. The two custom weights in particular have incomplete training-data, author, base-model, and license provenance.

All are blocked from production release until `models/manifest.json`, the corresponding model cards, and approval records satisfy [docs/models.md](docs/models.md). If rights cannot be established, remove the artifact from distributed releases and replace it with a properly authorized model.

## Runtime and development dependencies

The current direct runtime packages are NumPy, headless OpenCV, Pillow, Streamlit, PyTorch, TorchVision, and the headless Ultralytics distribution. Direct development tools include mypy, pip-audit, pre-commit, pytest/pytest-cov, and Ruff; the optional notebook group adds IPykernel, JupyterLab, KaggleHub, and Matplotlib. Python, `uv`, the container base image, operating-system packages, and all transitive packages also require inventory. Consult `pyproject.toml`, `uv.lock`, the final container SBOM, and each component's included metadata/license for the exact release set.

Do not copy dependency license text into this file from memory. A release process should:

1. generate an exact inventory from the locked production and development environments;
2. identify package, version, source, license expression, copyright notice, and distribution obligations;
3. resolve unknown, non-commercial, copyleft, or incompatible results;
4. include required notices/source offers/license texts with the distributed artifact;
5. archive the reviewed report with the release.

## Datasets, images, and fonts

The repository and container now ship no vehicle photographs. `images/README.md` is the only tracked file under `images/`; local Approved examples are ignored by Git and Docker. Evaluation data must remain in a separately controlled workflow with an exact file/dataset identity, source, license/permission, collection/privacy authority, transformations, retention, and deletion record.

The archived notebook references the [Moroccan Vehicle Registration Plates Kaggle listing](https://www.kaggle.com/datasets/elmehditaf96/moroccan-vehicle-registration-plates), which currently displays a CC0 Public Domain label. That listing documents the notebook's requested input only: no exact-file mapping proves that previously removed repository photographs or the model-training data came from it, and a listing label alone does not complete collection/privacy review. Do not use it as model, sample, or evaluation provenance without separately verified records.

Removing photographs from the current tree/container does not erase prior Git history, forks, clones, CI artifacts, or remote caches. Before public release, the owner must decide whether a coordinated history purge/cache cleanup is required. Do not perform a casual rewrite; preserve the decision and cleanup verification with the release record.

Notebook data, training/evaluation datasets, fonts, screenshots, and documentation media also require their own source and rights record. Do not assume that public availability grants training, redistribution, or commercial rights. Remove or replace anything whose provenance cannot be established.

## Reporting an omission

Report a suspected license/provenance omission privately when disclosure could expose confidential details; otherwise open an issue containing the component name, version/hash, source, and supporting license information. Do not attach restricted artifacts or personal data.
