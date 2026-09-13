"""Compatibility entry point for the maintained Privacy-GAN trainer.

Use ``python -m training.train_privacy_gan`` for new experiments.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_privacy_gan import main


if __name__ == "__main__":
    main()
