import zipfile

file_path = "C:/Users/CBNU/Desktop/이현지/24_2_NLP/Data/감정이태깅된자유대화/134-2.감정이 태깅된 자유대화 (청소년)/01-1.정식개방데이터/Validation/02.라벨링데이터/"
file_name = "VL_02.실외.zip"

try:
    with zipfile.ZipFile(file_path + file_name, 'r') as zip_ref:
        zip_ref.extractall("C:/Users/CBNU/Desktop/이현지/24_2_NLP/Data/A_Data/Validation/VL_02.실외")  # 추출될 폴더
    print("압축 해제 완료!")
except zipfile.BadZipFile:
    print("오류")
except Exception as e:
    print(f"오류: {e}")