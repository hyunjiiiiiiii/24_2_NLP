import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# KoBERT 모델 및 토크나이저 로드
MODEL_NAME = "monologg/kobert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, trust_remote_code=True)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 텍스트 데이터 읽기 함수
def load_text_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        texts = file.readlines()
    # 불필요한 공백 제거
    texts = [text.strip() for text in texts if text.strip()]
    return texts

# 감정 분석 함수
def analyze_sentiment(texts):
    model.eval()  # 평가 모드
    results = {"positive": 0, "neutral": 0, "negative": 0}
    detailed_results = []

    for text in texts:
        # 토큰화 및 텐서 변환
        encoding = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # 모델 추론
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # 클래스 매핑 (0: 긍정, 1: 중립, 2: 부정)
        if predicted_class == 0:
            results["positive"] += 1
            sentiment = "positive"
        elif predicted_class == 1:
            results["neutral"] += 1
            sentiment = "neutral"
        else:
            results["negative"] += 1
            sentiment = "negative"

        detailed_results.append({"text": text, "sentiment": sentiment})

    return results, detailed_results

# 재생성 조건 평가
def check_and_regenerate(texts):
    results, detailed_results = analyze_sentiment(texts)

    # 부정 비율 계산
    total_texts = len(texts)
    negative_ratio = results["negative"] / total_texts * 100

    print(f"Total texts: {total_texts}")
    print(f"Sentiment distribution: {results}")
    print(f"Negative sentiment ratio: {negative_ratio:.2f}%")

    # 부정 비율이 80% 이상일 경우 재생성
    if negative_ratio >= 80:
        print("Negative sentiment exceeds 80%. Regeneration needed!")
        regenerate_texts()
    else:
        print("Sentiment distribution is acceptable. No regeneration needed.")

    return detailed_results

# 재생성 로직 (예시)
def regenerate_texts():
    print("Regenerating texts...")
    # ChatGPT API 또는 다른 텍스트 생성 시스템 연동
    regenerated = ["새로운 동화 텍스트입니다. 긍정적인 내용이 많습니다!"]
    print(f"Generated texts: {regenerated}")

# 실행
if __name__ == "__main__":

    file_path = "./Data/100/Under_the_Sunlight_ko.txt"

    try:
        # 데이터 로드
        texts = load_text_data(file_path)
        print(f"Loaded {len(texts)} texts from {file_path}")

        # 감정 분석 및 재생성 확인
        detailed_results = check_and_regenerate(texts)

        # 상세 결과 출력
        print("Detailed Results:")
        for result in detailed_results:
            print(result)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
