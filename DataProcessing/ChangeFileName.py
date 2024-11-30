import os

# JSON 파일이 있는 폴더 경로
source_folder = "C:/Users/CBNU/Desktop/이현지/24_2_NLP/Data/A_Data/Validation/VL_02.실외"

# 파일 이름을 1부터 순차적으로 변경
def rename_files_in_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    files.sort()  # 파일 정렬
    
    for idx, file_name in enumerate(files, start=1):
        old_path = os.path.join(folder_path, file_name)
        new_name = f"{idx}.json"
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")

rename_files_in_folder(source_folder)