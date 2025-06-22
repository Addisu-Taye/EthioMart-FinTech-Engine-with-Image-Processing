import os
import logging
from pathlib import Path
import pandas as pd
from src.utils import load_conll, save_conll

logger = logging.getLogger(__name__)


class LabelingManager:
    def __init__(self, labeled_data_path="data/processed/labeled_data.conll"):
        self.labeled_data_path = Path(labeled_data_path)
        self.labeled_data_dir = self.labeled_data_path.parent
        self.labeled_data_dir.mkdir(parents=True, exist_ok=True)

    def has_labeled_data(self):
        """Check if labeled data exists and is valid"""
        if not self.labeled_data_path.exists():
            return False

        try:
            data = load_conll(self.labeled_data_path)
            return len(data) > 0
        except Exception as e:
            logger.warning(f"Invalid labeled data: {e}")
            return False

    def generate_initial_labels(self, raw_data_path="data/raw"):
        """Create pre-annotated labels from raw data"""
        try:
            raw_data_path = Path(raw_data_path)
            raw_files = list(raw_data_path.glob("*.csv"))
            if not raw_files:
                raise FileNotFoundError(f"No raw data files found in {raw_data_path}")

            # Sample messages for labeling
            sample_messages = []
            for file in raw_files[:5]:  # Use first 5 files
                df = pd.read_csv(file)
                if 'text' not in df.columns:
                    logger.warning(f"'text' column not found in {file.name}")
                    continue
                sample_messages.extend(df['text'].dropna().sample(n=min(10, len(df)), random_state=42).tolist())

            # Create initial CONLL format
            labeled_data = []
            for msg in sample_messages:
                tokens = msg.split()
                labels = ["O"] * len(tokens)
                labeled_data.append({"tokens": tokens, "labels": labels})

            # Save initial labels
            save_conll(labeled_data, self.labeled_data_path)
            logger.info(f"Generated initial labels at {self.labeled_data_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate initial labels: {e}")
            return False

    def launch_labeling_interface(self):
        """Start Label Studio interface"""
        try:
            import label_studio
            from label_studio import server
            os.environ['LABEL_STUDIO_USERNAME'] = 'admin@localhost'
            os.environ['LABEL_STUDIO_PASSWORD'] = 'password'
            logger.info("Starting Label Studio...")
            server.main()
            return True
        except ImportError:
            logger.error("Label Studio not installed. Run: pip install label-studio")
            return False
        except Exception as e:
            logger.error(f"Failed to start Label Studio: {e}")
            return False


if __name__ == "__main__":
    # Test the labeling manager
    logging.basicConfig(level=logging.INFO)
    labeler = LabelingManager()

    if not labeler.has_labeled_data():
        print("No labeled data found - generating samples...")
        labeler.generate_initial_labels()

    print("Launching labeling interface...")
    labeler.launch_labeling_interface()
