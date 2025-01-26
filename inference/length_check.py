import pandas as pd

def calculate_length_statistics(file_path):
    """
    데이터의 max_length 값과 평균 길이를 계산.
    """
    # 데이터 로드
    print("Loading data...")
    data = pd.read_csv(file_path, encoding="utf-8-sig")

    # 각 리뷰의 길이 계산
    print("Calculating lengths...")
    data['length'] = data['input'].apply(len)  # 'input' 컬럼의 문자열 길이 계산

    # 최대 길이와 평균 길이 계산
    max_length = data['length'].max()
    avg_length = data['length'].mean()

    print(f"Max Length: {max_length}")
    print(f"Average Length: {avg_length}")

    return max_length, avg_length

if __name__ == "__main__":
    # 경로 설정
    TEST_FILE = "../data/test.csv"

    # 길이 통계 계산
    max_len, avg_len = calculate_length_statistics(TEST_FILE)
