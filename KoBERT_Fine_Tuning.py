from transformers import AutoTokenizer, AutoModelForSequenceClassification

# KoBERT Tokenizer와 Model
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
model = AutoModelForSequenceClassification.from_pretrained("monologg/kobert", num_labels=3)  # 긍정, 부정, 중립
