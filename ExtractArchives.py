import os
import subprocess

def extract_and_delete_static_libraries(base_dir):
    """
    Search for .a static libraries in the given directory, extract .o files, and delete the original .a files.
    """
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".a"):
                file_path = os.path.join(root, file)
                print(f"[*] Found static library: {file_path}")
                
                # Create a directory to extract the .o files
                extract_dir = os.path.join(root, f"{file}_extracted")
                os.makedirs(extract_dir, exist_ok=True)
                
                try:
                    # Extract .o files using the 'ar' command
                    subprocess.run(["ar", "x", f"../{file}"], cwd=extract_dir, check=True)
                    print(f"[+] Extracted .o files to: {extract_dir}")
                    
                    # Delete the original .a file
                    os.remove(file_path)
                    print(f"[+] Deleted original static library: {file_path}")
                except subprocess.CalledProcessError as e:
                    print(f"[!] Failed to extract {file_path}: {e}")
                except Exception as e:
                    print(f"[!] Error processing {file_path}: {e}")

extract_and_delete_static_libraries("Binaries")