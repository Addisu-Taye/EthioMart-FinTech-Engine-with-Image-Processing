# EthioMart FinTech Engine

**Project:** Transforming Telegram Posts into a Smart FinTech Engine  
 
**Prepared by:** Addisu Taye  

---

## 1. Introduction  
This project aims to develop EthioMart's FinTech engine, which extracts structured business data from Telegram-based e-commerce channels and generates vendor lending scores.  

### Objectives:  
- Scrape and preprocess Telegram data (text & images).  
- Develop an Amharic Named Entity Recognition (NER) model.  
- Implement a vendor scoring system for micro-lending.  

---

## 2. Methodology  

### 2.1 Data Collection  
**Tools & Techniques:**  
- **Telegram API (`telethon`)** for automated scraping of Ethiopian e-commerce channels (e.g., ShagerOnlineStore).  
- Captures:  
  - Text messages  
  - Images (product photos, ads)  
  - Metadata (views, timestamps, sender info)  

**Challenges & Solutions:**  

| **Challenge**          | **Solution**                                      |
|------------------------|--------------------------------------------------|
| Rate limits            | Implemented throttling (1 request/sec)           |
| Data Cleaning          | Handled non-readable Unicode characters         |
| Amharic encoding       | Used UTF-8 normalization                         |
| Media storage          | Saved images in `data/raw/media/` with metadata  |

---

### 2.2 Data Preprocessing  
**Pipeline:**  

1. **Text Cleaning:**  
   - Normalized Ethiopic script (e.g., "ብር" → "birr").  
   - Removed duplicates, stopwords, and non-commercial posts.  

2. **Image Processing:**  
   - **OCR (Tesseract with Amharic support):** Extracted text from product images.  
   - **Product Detection (ViT Model):** Classified images into product categories.  

3. **Entity Extraction (Rule-Based):**  
   - **Regex patterns** for:  
     - Prices (`\d+\s*(ብር|ETB)`).  
     - Locations (e.g., "አዲስ አበባ" or "Addis Ababa").  

**Output:**  
Structured dataset in `data/processed/processed_data.csv`.  

---

### 2.3 Data Labeling for NER  
**Approach:** Semi-automated labeling with human review.  

**Tools:**  
1. **Label Studio** configured with 3 entity classes:  
   - `PRODUCT` (e.g., "ልብስ", "ስልክ").  
   - `PRICE` (e.g., "500 ብር").  
   - `LOC` (e.g., "መቀሌ").  

2. **Pre-Labeling with Rules:** Auto-tagged prices/locations using regex before manual review.  

**Dataset Stats:**  
- **50 labeled messages** (CoNLL format).  
- **Inter-annotator agreement:** 85% (Cohen's κ).  

---

### 2.4 Model Training  
**Pipeline:**  

1. **Model Selection:** Compared:  
   - `XLM-Roberta` (multilingual).  
   - `bert-tiny-amharic` (lightweight).  
   - `afroxlmr-large` (best for Amharic).  

2. **Fine-Tuning:**  
   - **Framework:** Hugging Face `Trainer`.  
   - **Best Model:** `afroxlmr-large` (F1: 0.84).  

---

### 2.5 Vendor Scoring System  
**Key Metrics:**  
1. **Engagement:**  
   - Avg. views/post.  
   - Posting frequency (posts/week).  

2. **Business Profile:**  
   - Avg. product price.  
   - Product variety (unique items).  

**Algorithm:**  
```python
def lending_score(vendor):
    score = (avg_views * 0.4) + (post_frequency * 0.3) + (avg_price * 0.2) + (product_variety * 0.1)
    return score * 100  # Scale to 0-100