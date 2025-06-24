import pandas as pd
import re
import os
import unicodedata
from pathlib import Path
import logging
import sys # Import sys to check stdout encoding

import spacy

# Assuming these are available in your project structure
# from src.utils import normalize_amharic
# from src.image_processing import ImageProcessor

# Placeholder for normalize_amharic if src.utils is not provided or accessible.
# You should replace this with your actual normalize_amharic implementation.
def normalize_amharic(text):
    """
    Placeholder for Amharic text normalization.
    Replace with your actual implementation from src.utils.
    """
    if pd.isna(text):
        return ""
    # Example basic normalization: replace common variant characters
    text = text.replace("ኅ", "ሕ").replace("ኧ", "አ").replace("ዐ", "አ")
    text = text.replace("ፀ", "ጸ").replace("ጐ", "ጎ").replace("ጒ", "ጉ")
    text = text.replace("ጏ", "ጓ").replace("ጘ", "ገ").replace("ጙ", "ግ")
    text = text.replace("ጚ", "ጋ").replace("ጛ", "ጌ").replace("ጜ", "ግ")
    text = text.replace("ሏ", "ለ").replace("ሟ", "መ").replace("ሯ", "ረ")
    text = text.replace("ሷ", "ሰ").replace("ሿ", "ሸ").replace("ቧ", "በ")
    text = text.replace("ቩ", "ቯ").replace("ፏ", "ፈ")
    return text


# Placeholder for ImageProcessor if src.image_processing is not provided or accessible.
# You should replace this with your actual ImageProcessor implementation.
class ImageProcessor:
    def __init__(self):
        logger.info("ImageProcessor initialized (placeholder).")
    
    def process_image(self, image_path):
        """
        Placeholder for image processing logic.
        Replace with your actual implementation from src.image_processing.
        """
        logger.info(f"Processing image: {image_path} (placeholder).")
        # In a real scenario, this would perform image operations
        return f"processed_{os.path.basename(image_path)}"


# Configure logging
# Ensure logging stream uses UTF-8 to prevent UnicodeEncodeError on some systems (e.g., Windows)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Direct logging to stdout
    ]
)
# Force stdout encoding if not already utf-8, primarily for Windows environments
# This helps in environments where default encoding might not be UTF-8 (e.g., Windows cmd)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception as e:
        logger.warning(f"Failed to reconfigure stdout to UTF-8: {e}")
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception as e:
        logger.warning(f"Failed to reconfigure stderr to UTF-8: {e}")

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        # NLP and image utilities
        # Ensure these are initialized correctly; they are not causing the AttributeError
        self.image_processor = ImageProcessor()
        # Multilingual NER model
        self.nlp = spacy.load("xx_ent_wiki_sm")

        # Patterns
        self.price_pattern = re.compile(
            r'(\d{1,3}(?:[\s,]?\d{3})*|\d+)\s*(ብር|ETB|Br|birr)', re.IGNORECASE
        )
        self.phone_pattern = re.compile(r'\b(?:09\d{8}|\+2519\d{8}|2519\d{8})\b')
        self.loc_patterns = [
            re.compile(r'(አዲስ\s*አበባ|Addis[\s-]?Ababa|AA)', re.IGNORECASE),
            re.compile(r'(መቀሌ|Mek[ae]lle)', re.IGNORECASE),
            re.compile(r'(ባህር\s*ዳር|Bahir[\s-]?Dar)', re.IGNORECASE),
            re.compile(r'(አዋሻ|Awash)', re.IGNORECASE),
            re.compile(r'(አማራ|Amhara)', re.IGNORECASE),
            re.compile(r'(አፋር|Afar)', re.IGNORECASE),
        ]

        # Regex patterns for image references
        self.image_url_pattern = re.compile(
            r'https?://[^\s/$.?#].[^\s]*?\.(?:jpg|jpeg|png|gif|bmp|svg|webp)(?:\?[^\s]*)?',
            re.IGNORECASE
        )
        self.markdown_image_pattern = re.compile(r'!\[.*?\]\s*\([^)]+\)', re.DOTALL)
        self.html_image_pattern = re.compile(r'<img[^>]+src=["\']?([^"\'>]+)["\']?[^>]*>', re.IGNORECASE)

        # Initialize cleaning counts
        self.cleaning_counts = {}


    def load_data(self, data_dir="data/raw"):
        """Load all CSV files except 'media.csv'."""
        dfs = []
        for file in Path(data_dir).glob("*.csv"):
            if file.stem.lower() != 'media':
                try:
                    df = pd.read_csv(file)
                    if 'text' not in df.columns:
                        logger.warning(f"Skipping {file.name}: no 'text' column")
                        continue
                    df['channel'] = file.stem
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file.name}: {e}")
        if not dfs:
            raise ValueError("No valid CSV files loaded from raw directory.")
        return pd.concat(dfs, ignore_index=True)

    def remove_emojis(self, text):
        """Remove emojis and pictographs using comprehensive Unicode ranges, and track counts."""
        if pd.isna(text):
            return ""
        original_text = str(text)
        
        # Expanded emoji pattern to cover more Unicode ranges for various symbols and emojis
        emoji_pattern = re.compile(
            "["
            # Emoticons
            u"\U0001F600-\U0001F64F"
            # Miscellaneous Symbols and Pictographs
            u"\U0001F300-\U0001F5FF"
            # Transport & Map Symbols
            u"\U0001F680-\U0001F6FF"
            # Flags (iOS)
            u"\U0001F1E0-\U0001F1FF"
            # Supplemental Symbols and Pictographs
            u"\U0001F900-\U0001F9FF" # Broader than just the subset U+1F926-U+1F937

            # Basic Multilingual Plane (BMP) symbols and dingbats
            u"\u2600-\u26FF"  # Miscellaneous Symbols (e.g., ⚡️)
            u"\u2700-\u27BF"  # Dingbats (e.g., ✔️)
            u"\u25A0-\u25FF"  # Geometric Shapes (e.g., 🔸, black square/circle)
            u"\u2B00-\u2BFF"  # Miscellaneous Symbols and Arrows
            u"\u2190-\u21FF"  # Arrows
            u"\u2300-\u23FF"  # Miscellaneous Technical
            u"\u2000-\u206F"  # General Punctuation (contains some symbols like bullet points, etc.)
            u"\u20D0-\u20FF"  # Combining Diacritical Marks for Symbols
            u"\u2100-\u214F"  # Letterlike Symbols, Number Forms

            # Specific common emoji/symbol-related characters
            u"\u200d"  # Zero Width Joiner (for complex emojis)
            u"\ufe0f"  # Variation Selector-16 (for emoji presentation)
            u"\u23cf"  # Eject symbol
            u"\u23e9"  # Fast-forward button
            u"\u231a"  # Watch symbol
            u"\u3030"  # Wavy Dash
            u"\u2B50"  # White Medium Star
            u"\u203C"  # Double Exclamation Mark
            u"\u2049"  # Question Exclamation Mark
            u"\u20E3"  # Combining Enclosing Keycap
            u"\u2122"  # Trade Mark Sign
            u"\u2139"  # Information Source

            # Other common symbol blocks that might contain emoji-like characters
            u"\U000024C2-\U0001F251" # A very broad range covering many enclosed alphanumerics, cjk, mahjong, playing cards etc.
            u"\U0001F000-\U0001F02F" # Mahjong Tiles
            u"\U0001F0A0-\U0001F0FF" # Playing Cards
            u"\U0001FA70-\U0001FAFF" # Symbols and Pictographs Extended-A

            # Catch-all for all supplementary planes, where most modern emojis are
            u"\U00010000-\U0010FFFF"
            "]+", flags=re.UNICODE
        )
        
        # Find all matches before substitution to count them
        matches = emoji_pattern.findall(original_text)
        if matches:
            if 'emojis_removed_count' not in self.cleaning_counts:
                self.cleaning_counts['emojis_removed_count'] = 0
            self.cleaning_counts['emojis_removed_count'] += len(matches)
        
        return emoji_pattern.sub('', original_text)

    def convert_amharic_numbers(self, text):
        """Convert Amharic numerals (፩፪...) to Arabic numerals."""
        if pd.isna(text):
            return ""
        amharic_to_arabic = {
            '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
            '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10',
            '፳': '20', '፴': '30', '፵': '40', '፶': '50',
            '፷': '60', '፸': '70', '፹': '80', '፺': '90',
            '፻': '100', '፼': '10000'
        }
        # Sort by length in descending order to handle '፲' before '፩', etc.
        for amh, arb in sorted(amharic_to_arabic.items(), key=lambda x: -len(x[0])):
            text = text.replace(amh, arb)
        return text

    def keep_only_relevant_chars(self, text):
        """
        Optionally remove everything except Amharic, English, digits, and punctuation.
        """
        if pd.isna(text):
            return ""
        # This regex now allows for a wider range of standard punctuation.
        # It explicitly includes '፡', '።', '፣', '፤', '፥', '፦', '፧', '፨' which are Amharic punctuation.
        return re.sub(r"[^\w\s\.\,\!\?\-\'\’\"፡።፣፤፥፦፧፨አ-፼]", "", text)

    def remove_image_references(self, text):
        """
        Removes common textual references to images (URLs, Markdown, HTML tags), and track counts.
        """
        if pd.isna(text):
            return ""
        original_text = str(text)
        current_text = original_text
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

        # Direct image URLs
        matches_url = self.image_url_pattern.findall(current_text)
        if matches_url:
            removed_this_call += len(matches_url)
            current_text = self.image_url_pattern.sub('', current_text)

        if removed_this_call > 0:
            if 'image_references_removed_count' not in self.cleaning_counts:
                self.cleaning_counts['image_references_removed_count'] = 0
            self.cleaning_counts['image_references_removed_count'] += removed_this_call
        
        return current_text

    def preprocess_text(self, text):
        """Clean and normalize text: emoji removal, numeral conversion, Amharic normalization, and image reference removal."""
        if pd.isna(text):
            return ""
        text = self.remove_image_references(text) # New step: remove image references first
        text = self.remove_emojis(text)
        text = unicodedata.normalize("NFKC", text)
        text = self.convert_amharic_numbers(text)
        text = normalize_amharic(text) # Assuming normalize_amharic handles its own normalization
        text = self.keep_only_relevant_chars(text)
        text = re.sub(r'\s+', ' ', text) # Normalize whitespace
        return text.strip()

    def extract_prices(self, text):
        """Extract numeric price values from text."""
        return [
            int(match.group(1).replace(',', '').replace(' ', ''))
            for match in self.price_pattern.finditer(text)
        ]

    def extract_locations(self, text):
        """Extract location mentions from text."""
        found = set()
        for pattern in self.loc_patterns:
            for match in pattern.finditer(text):
                found.add(match.group(0).strip())
        return list(found)

    def extract_phone_numbers(self, text):
        """Extract phone numbers from text."""
        if pd.isna(text):
            return []
        return self.phone_pattern.findall(str(text))

    def extract_named_entities(self, text):
        """Extract named entities (PER, ORG, PRODUCT) from text."""
        if pd.isna(text):
            return []
        doc = self.nlp(str(text))
        return [ent.text for ent in doc.ents if ent.label_ in {"PER", "ORG", "PRODUCT"}]


    def get_cleaning_summary(self):
        """Returns a dictionary with the counts of removed items."""
        return self.cleaning_counts

    def preprocess_all(self, output_file="data/processed/processed_data.csv"):
        """Run full pipeline: load, preprocess, extract features, save, and split outputs."""
        try:
            # Reset counts for a new run
            self.cleaning_counts = {
                'emojis_removed_count': 0,
                'image_references_removed_count': 0
            }

            df = self.load_data()

            # Text cleaning and feature extraction
            df['clean_text'] = df['text'].apply(self.preprocess_text)
            df['prices'] = df['clean_text'].apply(self.extract_prices)
            df['locations'] = df['clean_text'].apply(self.extract_locations)
            df['phones'] = df['clean_text'].apply(self.extract_phone_numbers)
            df['entities'] = df['clean_text'].apply(self.extract_named_entities)

            # Save main CSV
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"✅ Main preprocessing complete. Saved to {output_file}")

            # Split by channel
            for channel_name, group in df.groupby('channel'):
                path = f"data/processed/by_channel/{channel_name}.csv"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                group.to_csv(path, index=False)

            # Categorize by region (multiple names per region)
            region_map = {
                "Addis Ababa": ["አዲስ አበባ", "Addis Ababa", "AA"],
                "Mekelle": ["መቀሌ", "Mekelle", "Mekele"],
                "Awash": ["አዋሻ", "Awash"],
                "Amhara": ["አማራ", "Amhara","ባህር ዳር", "Bahir Dar"],
                "Afar": ["አፋር", "Afar"]
            }

            for region, aliases in region_map.items():
                # Ensure 'locations' column contains a list of strings for matching
                mask = df['locations'].apply(lambda locs: any(alias in locs for alias in aliases))
                region_df = df[mask]
                if not region_df.empty:
                    path = f"data/processed/by_region/{region.replace(' ', '_')}.csv"
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    region_df.to_csv(path, index=False)

            logger.info("✅ Split outputs by channel and grouped region complete.")
            
            # Log the cleaning summary
            summary = self.get_cleaning_summary()
            logger.info(f"📊 Data Cleaning Summary: {summary}")
            
            return df

        except Exception as e:
            # Changed the error symbol to a more common one to avoid encoding issues
            logger.exception("--- Preprocessing failed. Please check the traceback above for details.")
            raise e


# Script entry point
if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess_all()
