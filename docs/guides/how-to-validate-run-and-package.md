# How To Validate Run And Package

This guide covers the full local participant loop.

## Validate

Check entrypoint and stream contract:

```bash
cestim validate --estimator ./my-estimator/estimator.py
```

## Run Locally

Run local evaluation with the safer isolation boundary:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner subprocess
```

Use in-process runner for faster iteration during debugging:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner inprocess
```

## Run With Harness

Repository harness script:

```bash
./scripts/run-test-harness.sh quick
./scripts/run-test-harness.sh full
./scripts/run-test-harness.sh exhaustive
```

## Debug Mode

Use `--debug` to surface tracebacks and `--json` for structured JSON:

```bash
cestim run \
  --estimator ./my-estimator/estimator.py \
  --runner inprocess \
  --detail full \
  --profile \
  --debug \
  --json
```

## Package

Build a submission artifact:

```bash
cestim package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

Optional metadata files:

```bash
cestim package \
  --estimator ./my-estimator/estimator.py \
  --requirements ./my-estimator/requirements.txt \
  --submission-metadata ./my-estimator/submission.yaml \
  --approach ./my-estimator/APPROACH.md \
  --output ./submission.tar.gz
```

## Submit

`TODO: hosted AIcrowd submission upload flow is not implemented yet.`
