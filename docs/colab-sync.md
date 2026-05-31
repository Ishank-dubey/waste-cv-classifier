# Colab Sync Notes

## Recommended Workflow

1. Open the notebook from GitHub using the Colab badge in the README.
2. Run the setup cells.
3. Save experiment outputs to Google Drive or download them locally.
4. Commit only code, notebooks, and small documentation files.
5. Do not commit datasets, trained model weights, or patent-sensitive implementation details.

## Pulling Latest Code in Colab

```bash
%cd /content/waste-cv-classifier
!git pull
```

## Pushing Changes from Colab

Pushing directly from Colab requires GitHub authentication. For now, the safer workflow is:

1. Edit locally or download changed notebooks.
2. Review changes.
3. Commit and push from the local machine.
