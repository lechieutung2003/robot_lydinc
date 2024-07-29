import os
import unidecode

def remove_vietnamese_accents(text):
    return unidecode.unidecode(text)

def rename_files_and_folders(root_path):
    for root, dirs, files in os.walk(root_path, topdown=False):
        # Đổi tên tệp
        for name in files:
            new_name = remove_vietnamese_accents(name)
            if new_name != name:
                os.rename(os.path.join(root, name), os.path.join(root, new_name))
        
        # Đổi tên thư mục
        for name in dirs:
            new_name = remove_vietnamese_accents(name)
            if new_name != name:
                os.rename(os.path.join(root, name), os.path.join(root, new_name))

# Gọi hàm với đường dẫn gốc của thư mục cần xử lý
rename_files_and_folders("F:\\LYDINC\\AI\\robot_lydinc\\Data processing\\origin")