import zipfile

file_path = "C:/Users/CBNU/Downloads/015.동화 줄거리 생성 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/"
file_name = "VL_05T_신체운동_건강_03S_초등_고학년.zip"

try:
    with zipfile.ZipFile(file_path + file_name, 'r') as zip_ref:
        zip_ref.extractall("C:/Users/CBNU/Downloads/Validation/VL_05T_신체운동_건강_03S_초등_고학년")  # 추출될 폴더
    print("압축 해제 완료!")
except zipfile.BadZipFile:
    print("오류")
except Exception as e:
    print(f"오류: {e}")