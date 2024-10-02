from PIL import Image
import pillow_heif

def convert_heic_to_png(heic_path, png_path):
    # Read HEIC file
    heif_file = pillow_heif.read_heif(heic_path)
    
    # Convert HEIC to PIL Image
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    
    # Save as PNG
    image.save(png_path, "PNG")

# Example usage
heic_path = "D:\\Artificial Intelligence\\article\\Code\\image\\IMG_2581.HEIC"
png_path = "D:\\Artificial Intelligence\\article\\Code\\image\\IMG_2581.png"
convert_heic_to_png(heic_path, png_path)