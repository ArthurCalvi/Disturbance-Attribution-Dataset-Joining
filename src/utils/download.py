import gdown
import argparse
import os
import logging
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_unzip_gdrive_file(url, output_path):
    """
    Downloads a file from Google Drive and unzips it if it is a zip file.

    Args:
        url (str): The Google Drive file URL.
        output_path (str): The path to the directory where the contents will be extracted.
    """
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            logging.info(f"Created directory: {output_path}")

        logging.info(f"Starting download from {url}")
        
        # gdown.download will save the file in the output_path directory
        # and return the full path to the downloaded file.
        downloaded_file_path = gdown.download(url, output=output_path, quiet=False, use_cookies=False, fuzzy=True)

        if downloaded_file_path and os.path.exists(downloaded_file_path):
            logging.info(f"Download completed successfully. File saved at {downloaded_file_path}")

            if downloaded_file_path.endswith('.zip'):
                logging.info(f"Unzipping {downloaded_file_path} to {output_path}")
                with zipfile.ZipFile(downloaded_file_path, 'r') as zip_ref:
                    zip_ref.extractall(output_path)
                logging.info("Unzipping completed successfully.")

                # Clean up the zip file
                os.remove(downloaded_file_path)
                logging.info(f"Removed temporary file: {downloaded_file_path}")
            else:
                logging.info("Downloaded file is not a zip file. No extraction needed.")
        else:
            logging.error(f"Download failed. File could not be downloaded to {output_path}.")
            raise Exception("gdown download failed")


    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download a file from Google Drive and unzip it.")
    parser.add_argument("--url", type=str, required=True, help="Google Drive file URL.")
    parser.add_argument("--output", type=str, required=True, help="Path to save and extract the downloaded file.")
    
    args = parser.parse_args()
    
    download_and_unzip_gdrive_file(args.url, args.output) 