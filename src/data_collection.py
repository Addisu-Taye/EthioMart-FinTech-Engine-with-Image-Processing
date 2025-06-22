import os
import sys
from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.tl.types import MessageMediaPhoto
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('telegram_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TelegramScraper:
    def __init__(self, api_id=None, api_hash=None):
        """Initialize with hardcoded Ethiopian e-commerce channels"""
        load_dotenv()
        
        self.api_id = api_id or os.getenv('TELEGRAM_API_ID')
        self.api_hash = api_hash or os.getenv('TELEGRAM_API_HASH')
        
        # Hardcoded Ethiopian e-commerce channels
        self.channels = [
           # '@shagerstore',    # Example: Shager Online Store
            #'@addismart',      # Example: AddisMart
            'ZemenExpress',
            'nevacomputer',
            'meneshayeofficial', 
            'ethio_brand_collection',
            'Leyueqa',
            'sinayelj',
            'Shewabrand',
            'helloomarketethiopia',
            'modernshoppingcenter',
            'qnashcom',
            'Fashiontera',
            'kuruwear',
            'gebeyaadama',
            'MerttEka',
            'forfreemarket',
            'classybrands',
            'marakibrand',
            'aradabrand2',
            'marakisat2',
            'belaclassic',
            'AwasMart'
               ]
        
        if not self.api_id or not self.api_hash:
            error_msg = "API credentials missing. Please set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            self.client = TelegramClient('ethiomart_session', self.api_id, self.api_hash)
            logger.info("Telegram client initialized with hardcoded Ethiopian channels")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram client: {str(e)}")
            raise

    async def scrape_channel(self, channel_name, limit=1000):
        """Scrape messages from a single Ethiopian e-commerce channel"""
        messages = []
        try:
            async with self.client:
                channel = await self.client.get_entity(channel_name)
                async for message in self.client.iter_messages(channel, limit=limit):
                    try:
                        msg_data = {
                            'id': message.id,
                            'date': message.date,
                            'views': message.views or 0,
                            'text': message.text,
                            'media': bool(message.media),
                            'channel': channel_name,
                            'timestamp': datetime.now()
                        }

                        if isinstance(message.media, MessageMediaPhoto):
                            media_dir = f"data/raw/media/{channel_name}"
                            os.makedirs(media_dir, exist_ok=True)
                            media_path = f"{media_dir}/{message.id}.jpg"
                            await message.download_media(media_path)
                            msg_data['media_path'] = media_path

                        messages.append(msg_data)

                    except Exception as e:
                        logger.warning(f"Error processing message {message.id}: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Failed to scrape Ethiopian channel {channel_name}: {str(e)}")
            raise

        return pd.DataFrame(messages)

    def scrape_all(self, output_dir="data/raw"):
        """Scrape all hardcoded Ethiopian e-commerce channels"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f"{output_dir}/media", exist_ok=True)

            with self.client:
                for channel in self.channels:
                    try:
                        logger.info(f"Scraping Ethiopian channel: {channel}...")
                        df = self.client.loop.run_until_complete(
                            self.scrape_channel(channel)
                        )

                        output_file = f"{output_dir}/{channel.replace('@', '')}.csv"
                        df.to_csv(output_file, index=False)
                        logger.info(f"Saved {len(df)} messages from {channel} to {output_file}")

                    except Exception as e:
                        logger.error(f"Failed to scrape Ethiopian channel {channel}: {str(e)}")
                        continue

            logger.info("Finished scraping all hardcoded Ethiopian e-commerce channels")

        except Exception as e:
            logger.error(f"Fatal error in scrape_all: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        if not load_dotenv():
            logger.warning(".env file not found or empty")

        scraper = TelegramScraper()
        scraper.scrape_all()

    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)