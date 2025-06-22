import pandas as pd
import re
import os
import unicodedata
from pathlib import Path
import logging
import sys  # Import sys to check stdout encoding
import spacy

# Placeholder for normalize_amharic function
def normalize_amharic(text):
    """
    Placeholder for Amharic text normalization.
    Replace with your actual implementation from src.utils if available.
    """
    if pd.isna(text):
        return ""
    # Basic normalization for common Amharic character variants
    text = text.replace("ኅ", "ሕ").replace("ኧ", "አ").replace("ዐ", "አ")
    text = text.replace("ፀ", "ጸ").replace("ጐ", "ጎ").replace("ጒ", "ጉ")
    text = text.replace("ጏ", "ጓ").replace("ጘ", "ገ").replace("ጙ", "ግ")
    text = text.replace("ጚ", "ጋ").replace("ጛ", "ጌ").replace("ጜ", "ግ")
    text = text.replace("ሏ", "ለ").replace("ሟ", "መ").replace("ሯ", "ረ")
    text = text.replace("ሷ", "ሰ").replace("ሿ", "ሸ").replace("ቧ", "በ")
    text = text.replace("ቩ", "ቯ").replace("ፏ", "ፈ")
    return text


# Placeholder for ImageProcessor class
class ImageProcessor:
    def __init__(self):
        pass

    def process_image(self, image_path):
        """
        Placeholder for image processing logic.
        Replace with your actual implementation from src.image_processing if available.
        """
        logger.info(f"Processing image: {image_path} (placeholder).")
        return f"processed_{os.path.basename(image_path)}"


# Configure logging to use UTF-8 encoding for stream handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        print("DataPreprocessor: Initializing class instance.")
        self.image_processor = ImageProcessor()  # Placeholder or actual ImageProcessor
        self.nlp = spacy.load("xx_ent_wiki_sm")  # Multilingual NER model

        # Patterns for extraction
        self.price_pattern = re.compile(r'(\d{1,3}(?:[\s,]?\d{3})*|\d+)\s*(ብር|ETB|Br|birr)', re.IGNORECASE)
        self.phone_pattern = re.compile(r'\b(?:09\d{8}|\+2519\d{8}|2519\d{8})\b')
        self.loc_patterns = [
            re.compile(r'(አዲስ\s*አበባ|Addis[\s-]?Ababa|AA)', re.IGNORECASE),
            re.compile(r'(መቀሌ|Mek[ae]lle)', re.IGNORECASE),
            re.compile(r'(ባህር\s*ዳር|Bahir[\s-]?Dar)', re.IGNORECASE),
            re.compile(r'(አዋሻ|Awash)', re.IGNORECASE),
            re.compile(r'(አማራ|Amhara)', re.IGNORECASE),
            re.compile(r'(አፋር|Afar)', re.IGNORECASE),
        ]

        # Regex patterns for image detection
        self.image_url_pattern = re.compile(
            r'https?://[^\s/$.?#].[^\s]*?\.(?:jpg|jpeg|png|gif|bmp|svg|webp)(?:\?[^\s]*)?',
            re.IGNORECASE
        )
        self.markdown_image_pattern = re.compile(r'!\[.*?\]\s*$[^)]+$', re.DOTALL)
        self.html_image_pattern = re.compile(r'<img[^>]+src=["\']?([^"\'>]+)["\']?[^>]*>', re.IGNORECASE)

        self.cleaning_counts = {}

    def load_data(self, data_dir="data/raw"):
        dfs = []
        for file in Path(data_dir).glob("*.csv"):
            if file.stem.lower() != 'media':
                try:
                    df = pd.read_csv(file)
                    if 'text' not in df.columns:
                        logger.warning(f"Skipping {file.name}: no 'text' column found.")
                        continue
                    df['channel'] = file.stem
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file.name}: {e}")
        if not dfs:
            raise ValueError(f"No valid CSV files with a 'text' column loaded from {data_dir}.")
        return pd.concat(dfs, ignore_index=True)

    def remove_emojis(self, text):
        if pd.isna(text):
            return ""
        original_text = str(text)

        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # Transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols
            u"\U0001FA00-\U0001FA6F"  # Chess symbols
            u"\U0001F700-\U0001F77F"  # Alchemical symbols
            u"\u2600-\u26FF"          # ⚡️ and other misc symbols
            u"\u2700-\u27BF"          # Dingbats
            u"\u25A0-\u25FF"          # Geometric shapes
            u"\u2B00-\u2BFF"          # Arrows & shapes
            u"\u2190-\u21FF"          # Arrows
            u"\u2000-\u206F"          # General punctuation
            u"\u20D0-\u20FF"          # Combining marks
            u"\u2100-\u214F"          # Letterlike symbols
            u"\u200d"                 # Zero width joiner
            u"\ufe0f"                 # Variation selector
            u"\u23cf"                 # Eject symbol
            u"\u23e9"                 # Fast-forward
            u"\u231a"                 # Watch
            u"\u3030"                 # Wavy dash
            u"\u2B50"                 # Star
            u"\u203C"                 # Double exclamation
            u"\u2049"                 # Interrobang
            u"\u20E3"                 # Enclosing keycap
            u"\U00010000-\U0010FFFF"  # High plane Unicode
            "]+", flags=re.UNICODE
        )

        matches = emoji_pattern.findall(original_text)
        if matches:
            if 'emojis_removed_count' not in self.cleaning_counts:
                self.cleaning_counts['emojis_removed_count'] = 0
            self.cleaning_counts['emojis_removed_count'] += len(matches)
        return emoji_pattern.sub('', original_text)

    def convert_amharic_numbers(self, text):
        amharic_to_arabic = {
            '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
            '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10',
            '፳': '20', '፴': '30', '፵': '40', '፶': '50',
            '፷': '60', '፸': '70', '፹': '80', '፺': '90',
            '፻': '100', '፼': '10000'
        }
        for amh, arb in sorted(amharic_to_arabic.items(), key=lambda x: -len(x[0])):
            text = text.replace(amh, arb)
        return text

    def keep_only_relevant_chars(self, text):
        if pd.isna(text):
            return ""
        return re.sub(r"[^a-zA-Z0-9አ-፼\s]", "", text)

    def remove_image_references(self, text):
        if pd.isna(text):
            return ""
        current_text = str(text)
        removed_this_call = 0

        # Markdown images
        matches_md = self.markdown_image_pattern.findall(current_text)
        if matches_md:
            removed_this_call += len(matches_md)
            current_text = self.markdown_image_pattern.sub('', current_text)

        # HTML images
        matches_html = self.html_image_pattern.findall(current_text)
        if matches_html:
            removed_this_call += len(matches_html)
            current_text = self.html_image_pattern.sub('', current_text)

        # URLs ending in image extensions
        matches_url = self.image_url_pattern.findall(current_text)
        if matches_url:
            removed_this_call += len(matches_url)
            current_text = self.image_url_pattern.sub('', current_text)

        if removed_this_call > 0:
            self.cleaning_counts.setdefault('image_references_removed_count', 0)
            self.cleaning_counts['image_references_removed_count'] += removed_this_call

        return current_text.strip()

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = self.remove_image_references(text)
        text = self.remove_emojis(text)
        text = unicodedata.normalize("NFKC", text)
        text = self.convert_amharic_numbers(text)
        text = normalize_amharic(text)
        text = self.keep_only_relevant_chars(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_prices(self, text):
        return [
            int(match.group(1).replace(',', '').replace(' ', ''))
            for match in self.price_pattern.finditer(text)
        ]

    def extract_locations(self, text):
        found = set()
        for pattern in self.loc_patterns:
            for match in pattern.finditer(text):
                found.add(match.group(0).strip())
        return list(found)

    def extract_phone_numbers(self, text):
        if pd.isna(text):
            return []
        return self.phone_pattern.findall(str(text))

    def extract_named_entities(self, text):
        if pd.isna(text):
            return []
        doc = self.nlp(str(text))
        return [ent.text for ent in doc.ents if ent.label_ in {"PER", "ORG", "PRODUCT"}]

    def get_cleaning_summary(self):
        return self.cleaning_counts

    def preprocess_all(self, output_file="data/processed/processed_data.csv"):
        try:
            self.cleaning_counts = {
                'emojis_removed_count': 0,
                'image_references_removed_count': 0
            }

            df = self.load_data()
            logger.info("Starting text cleaning and feature extraction...")
            df['clean_text'] = df['text'].apply(self.preprocess_text)
            df['prices'] = df['clean_text'].apply(self.extract_prices)
            df['locations'] = df['clean_text'].apply(self.extract_locations)
            df['phones'] = df['clean_text'].apply(self.extract_phone_numbers)
            df['entities'] = df['clean_text'].apply(self.extract_named_entities)

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"Main preprocessing complete. Saved to {output_file}")

            logger.info("Splitting data by channel...")
            for channel_name, group in df.groupby('channel'):
                path = f"data/processed/by_channel/{channel_name}.csv"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                group.to_csv(path, index=False)
                logger.info(f"Saved channel '{channel_name}' to {path}")

            logger.info("Categorizing data by region...")
            region_map = {
                "Addis Ababa": ["አዲስ አበባ", "Addis Ababa", "AA"],
                "Mekelle": ["መቀሌ", "Mekelle", "Mekele"],
                "Awash": ["አዋሻ", "Awash"],
                "Amhara": ["አማራ", "Amhara", "ባህር ዳር", "Bahir Dar"],
                "Afar": ["አፋር", "Afar"]
            }

            for region, aliases in region_map.items():
                mask = df['locations'].apply(lambda locs: any(alias in locs for alias in aliases))
                region_df = df[mask]
                if not region_df.empty:
                    path = f"data/processed/by_region/{region.replace(' ', '_')}.csv"
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    region_df.to_csv(path, index=False)
                    logger.info(f"Saved region '{region}' to {path}")
                else:
                    logger.info(f"No data found for region '{region}'. Skipping save.")

            logger.info("Split outputs by channel and grouped region complete.")
            summary = self.get_cleaning_summary()
            logger.info(f"Data Cleaning Summary: {summary}")
            return df
        except Exception as e:
            logger.exception("An error occurred during preprocessing.")
            raise e


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess_all()