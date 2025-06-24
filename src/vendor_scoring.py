import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class VendorScorer:
    def __init__(self, model_path="data/models/ner_model"):
        self.model_path = model_path
        self.image_processor = ImageProcessor()
    
    def calculate_vendor_metrics(self, vendor_df):
        """Calculate key metrics for vendor assessment"""
        if len(vendor_df) == 0:
            return None
            
        # Activity & Consistency
        posting_freq = self._calculate_posting_frequency(vendor_df)
        
        # Market Reach & Engagement
        avg_views = vendor_df['views'].mean()
        top_post = vendor_df.loc[vendor_df['views'].idxmax()]
        
        # Business Profile
        avg_price = self._calculate_average_price(vendor_df)
        product_variety = self._count_unique_products(vendor_df)
        
        # Calculate lending score (weighted formula)
        lending_score = (avg_views * 0.4 + 
                        posting_freq * 0.3 + 
                        avg_price * 0.2 + 
                        product_variety * 0.1)
        
        return {
            'vendor_name': vendor_df['channel'].iloc[0],
            'avg_views_per_post': round(avg_views, 1),
            'posts_per_week': round(posting_freq, 1),
            'avg_price_etb': round(avg_price, 2),
            'product_variety': product_variety,
            'top_post_views': top_post['views'],
            'top_post_product': self._extract_products(top_post['text']),
            'lending_score': round(lending_score, 2),
            'last_updated': datetime.now().strftime("%Y-%m-%d")
        }
    
    def _calculate_posting_frequency(self, df):
        """Calculate average posts per week"""
        if 'date' not in df.columns:
            return 0
            
        df['date'] = pd.to_datetime(df['date'])
        time_span = (df['date'].max() - df['date'].min()).days
        if time_span == 0:
            return len(df)
        return (len(df) / time_span) * 7
    
    def _calculate_average_price(self, df):
        """Calculate average product price"""
        prices = []
        for text in df['text']:
            prices.extend(self._extract_prices(text))
        return np.mean(prices) if prices else 0
    
    def _count_unique_products(self, df):
        """Count unique products mentioned"""
        products = set()
        for text in df['text']:
            products.update(self._extract_products(text))
        return len(products)
    
    def _extract_prices(self, text):
        """Extract prices from text"""
        if pd.isna(text):
            return []
        matches = re.findall(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:ብር|ETB|Br|birr)', str(text))
        return [float(match.replace(',', '')) for match in matches]
    
    def _extract_products(self, text):
        """Extract products from text (simplified)"""
        if pd.isna(text):
            return []
        # This would normally use the NER model
        return list(set(re.findall(r'[\w-]+ ልብስ|[\w-]+ ሸሚዝ|[\w-]+ እቃ', str(text))))