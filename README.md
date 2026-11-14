# Reproducibility details (MemarNegar)

## Temporal cut-off (anchor) commits
| System | Cut-off datetime | Anchor commit (snapshot for static metrics + dependency graph; co-change = commits reachable from it) |
|---|---|---|
| Spring Framework | 2020-10-31 23:59:59 | 7be7e5beb4e7d808761f06e386b359d45a0890d8 |
| Elasticsearch | 2021-09-30 23:59:59 | eed2c816e4b95ff2ed8b64fd341f5b3161d3d533 |
| Hadoop | 2018-07-31 23:59:59 | 4b540bbfcf02d828052999215c6135603d98f5db |

Target window = commits in `anchor..HEAD`, merge commits excluded.
Repository HEAD (end of the label window) used for each system, from `git rev-parse HEAD`:
- Spring Framework: 8d48a1e34b4ef1a8edc6b3b5c8ecf0ec2c70ec61
- Elasticsearch: c662590b6d51592eeac1bc8b66793659ac735880
- Hadoop: 7d474d37fe353a5dc05a6bfc4f1a80c5094eda99

## Environment
Python 3.12.2; scikit-learn 1.7.2; XGBoost 3.1.2; NetworkX 3.6; javalang 0.13.0; GitPython 3.1.45; pandas 2.3.3; NumPy 2.3.5; SciPy 1.16.3.

## File -> class mapping
- One class entity = one `.java` file, identified by its path converted to `package.ClassName`.
- Static metrics use the first type declaration in the file.
- Change counts are per file over non-merge commits (`git log -M --name-status --no-merges`); renamed files are attributed to their final path.

## Baselines and models
- Baselines: `DummyClassifier(strategy="most_frequent")` and `DummyClassifier(strategy="stratified")`, random_state=42.
- Main split: stratified 80/20, seed 42. Tuning: grid search, 5-fold CV, balanced accuracy.
- XGBoost grid: n_estimators {100,200}, max_depth {3,5}, learning_rate {0.01,0.1}. SVM grid: C {0.1,1}, kernel rbf. Random forest grid: n_estimators {100,200}, max_depth {10,15}, min_samples_split {5,10}. GaussianNB grid: var_smoothing {1e-9,1e-8}.
