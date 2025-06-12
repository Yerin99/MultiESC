import json
import os
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK 리소스 다운로드
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# NRC-VAD 사전 로드
def load_nrc_vad(path='./MultiESC/data/NRC-VAD-Lexicon-v2.1.txt'):
    vad_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sp_line = line.split()
            if len(sp_line) > 4:
                continue
            vad_dict[sp_line[0].strip().lower()] = sp_line[1:]
    print(f"NRC-VAD 사전 크기: {len(vad_dict)}개 단어")
    return vad_dict

# 텍스트 정규화
def normalize_text(text):
    if not text:
        return ""
    # 소문자로 변환 및 특수문자 제거
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # 여러 공백을 하나로 변환
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 데이터셋 로드 및 텍스트 추출
def load_dataset(file_path):
    utterances = []
    word_count = Counter()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    dialog = data.get('dialog', [])
                    
                    # 대화에서 텍스트 추출
                    for turn in dialog:
                        text = turn.get('text', '')
                        if text:
                            normalized_text = normalize_text(text)
                            utterances.append(normalized_text)
                            
                            # 단어 분리 및 카운트
                            words = word_tokenize(normalized_text)
                            for word in words:
                                if word.isalpha():  # 숫자와 특수문자 제외
                                    word_count[word] += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {str(e)}")
    
    return utterances, word_count

# OOV 분석
def analyze_oov(word_count, vad_dict):
    stop_words = set(stopwords.words('english'))
    
    # 불용어 제외한 모든 단어
    all_words = set([word for word in word_count.keys() if word not in stop_words])
    all_words_count = sum([count for word, count in word_count.items() if word not in stop_words])
    
    # OOV 단어 (VAD 사전에 없는 단어)
    oov_words = set([word for word in all_words if word not in vad_dict])
    oov_count = sum([word_count[word] for word in oov_words])
    
    # 통계 계산
    vocab_size = len(all_words)
    oov_size = len(oov_words)
    oov_ratio = oov_size / vocab_size if vocab_size > 0 else 0
    
    # 빈도 기반 OOV 비율
    oov_freq_ratio = oov_count / all_words_count if all_words_count > 0 else 0
    
    # 상위 N개 OOV 단어와 빈도
    top_oov_words = sorted([(word, word_count[word]) for word in oov_words], 
                          key=lambda x: x[1], reverse=True)[:20]
    
    return {
        'vocab_size': vocab_size,
        'oov_size': oov_size,
        'oov_ratio': oov_ratio,
        'oov_freq_ratio': oov_freq_ratio,
        'top_oov_words': top_oov_words
    }

def main():
    print("NRC-VAD Lexicon v2.1과 ESConv 데이터셋 간의 OOV 분석")
    print("-" * 60)
    
    # NRC-VAD 사전 로드
    vad_dict = load_nrc_vad()
    
    # 데이터셋 파일 경로
    dataset_files = [
        './MultiESC/data/train.txt',
        './MultiESC/data/test.txt',
        './MultiESC/data/valid.txt'
    ]
    
    # 각 데이터셋 파일에 대한 OOV 분석
    for file_path in dataset_files:
        print(f"\n{os.path.basename(file_path)} 파일 분석 중...")
        
        # 데이터셋 로드
        utterances, word_count = load_dataset(file_path)
        print(f"총 {len(utterances)}개 발화, {sum(word_count.values())}개 단어 발견")
        
        # OOV 분석
        result = analyze_oov(word_count, vad_dict)
        
        # 결과 출력
        print(f"어휘 크기: {result['vocab_size']}개 단어")
        print(f"OOV 단어 수: {result['oov_size']}개")
        print(f"OOV 비율 (단어 기준): {result['oov_ratio']:.2%}")
        print(f"OOV 비율 (빈도 기준): {result['oov_freq_ratio']:.2%}")
        
        print("\n상위 OOV 단어 (빈도순):")
        for word, count in result['top_oov_words']:
            print(f"  {word}: {count}회")

if __name__ == "__main__":
    main() 