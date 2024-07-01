import os


def print_directory_contents(path, indent=""):
    """
    이 함수는 주어진 경로(path)의 모든 하위 폴더와 파일 구조를 출력합니다.
    """
    # 현재 경로의 모든 파일과 폴더 리스트를 가져옵니다.
    contents = os.listdir(path)

    for item in contents:
        full_path = os.path.join(path, item)
        # 만약 현재 아이템이 폴더라면 재귀적으로 호출하여 하위 폴더와 파일을 출력합니다.
        if os.path.isdir(full_path):
            if item == "output" or item == ".git":
                continue  # "result" 폴더나 "old" 폴더는 건너뜁니다.
            print(indent + "Folder:", item)
            print_directory_contents(
                full_path, indent + "    "
            )  # 재귀 호출 시 indent를 수정하여 들여쓰기를 제공합니다.
        else:
            print(indent + "File:", item)


# 현재 디렉토리를 기준으로 시작합니다.
start_path = os.getcwd()
print("Starting directory:", start_path)
print_directory_contents(start_path)
