import os
import urllib.request
import urllib.error

def load_font_names(config_path):
    """Read font names from the text configuration file"""
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found. Please make sure the file exists.")
        return []
        
    with open(config_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def download_google_fonts(font_names, output_dir="./fonts"):
    """Automatically download available .ttf files from Google Fonts GitHub"""
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "https://raw.githubusercontent.com/google/fonts/main/ofl"
    styles = ["Regular", "Medium", "Bold"]
    
    downloaded_count = 0
    
    for font in font_names:
        clean_name = font.replace(" ", "").lower()
        file_prefix = font.replace(" ", "")
        success = False
        
        for style in styles:
            file_name = f"{file_prefix}-{style}.ttf"
            download_url = f"{base_url}/{clean_name}/{file_name}"
            save_path = os.path.join(output_dir, f"{file_prefix}.ttf")
            
            try:
                urllib.request.urlretrieve(download_url, save_path)
                print(f"Successfully downloaded: {font} -> {save_path}")
                success = True
                downloaded_count += 1
                break
            except urllib.error.HTTPError:
                continue
                
        if not success:
            print(f"Skipped: {font} (Not a Google Font or requires manual copy)")
            
    print(f"\nDownload task finished. Successfully downloaded {downloaded_count} fonts.")

if __name__ == "__main__":
    # Path to your configuration file
    CONFIG_FILE = "utils/fonts_list.txt"
    FONT_DIRECTORY = "./fonts"
    
    # Step 1: Load fonts from your txt file
    fonts_to_download = load_font_names(CONFIG_FILE)
    
    # Step 2: Start downloading
    if fonts_to_download:
        print(f"Loaded {len(fonts_to_download)} fonts from {CONFIG_FILE}. Starting download...")
        download_google_fonts(fonts_to_download, FONT_DIRECTORY)