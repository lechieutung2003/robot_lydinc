import os
def get_all_images(root_folder):
    # Input: Đường dẫn đến thư mục gốc
    # Output: Đường dẫn các hình ảnh

    image_files = []
    COUNT = 0

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
                COUNT += 1

    print(COUNT, "images")

    return image_files

def write_path_images(root, fileName):
    # Input: 
    #       Đường dẫn đến thư mục gốc
    #       Tên tệp để lưu các đường dẫn

    images = get_all_images(root)

    with open(fileName, 'w', encoding="utf-8") as file:
        for image in images:
            file.write(f"{image}\n")

def resize_image(in_path, out_path, width, height):
    from PIL import Image

    img = Image.open(in_path)

    img = img.resize((width, height))

    img.save(out_path)

    print("Size changed...")

def copy_image(source_path, destination_folder):
    import shutil
    # Sao chép ảnh
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
        # return print(f"{destination_folder} is not exist")
    shutil.copy(source_path, destination_folder)

def divide_image(root):
    images = get_all_images(root)
    positive = []
    negative = []
    for img in images:
        img_name = os.path.split(img)[-1].split(".")[0]
        img_path = os.path.split(img)[0]
        folders = img_path.split("\\")
        folder_IQ = folders.pop(-1)
        folder_PT = folders.pop(-1)
        
        if (len(img_name.split("_")) < 2):
            continue
        EM = img_name.split("_")[2]

        if EM == "00":
            copy_image(img, f"emotion/00/{folder_PT}/{folder_IQ}")
        elif EM == "01" or EM == "1":
            copy_image(img, f"emotion/01/{folder_PT}/d{folder_IQ}")
        elif EM == "02" or EM == "2":
            copy_image(img, f"emotion/02/{folder_PT}/{folder_IQ}")
        else:
            copy_image(img, f"emotion/03")

