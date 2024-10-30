"""Module for loading data using pandas and/or Hugging Face datasets / HuggingFace hub.

Make sure to log in Hugging Face

```bash
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN --add-to-git-credential
```
"""

import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


def load_hf_dataset():
    """Load a dataset from the Hugging Face hub."""
    pass
