import queue
import threading
import time
import ftplib
import os
import shutil
import zipfile
import pandas as pd
import re
import argparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
# Use tqdm.auto to fallback if widgets are missing/broken
from tqdm.auto import tqdm

# Load environment variables (FTP_HOST, FTP_USER, FTP_PASS)
load_dotenv()

# --- Utility Functions ---

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    If the error is due to an access error (read only file), it attempts to add write permission and then retries.
    """
    import stat
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        try:
            func(path)
            return
        except Exception:
            pass
    
    # print(f"Warning: Could not delete {path}. Retrying in 1s...")
    time.sleep(1)
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"Failed to force delete {path}: {e}")

def robust_rmtree(path, retries=5, delay=1.0):
    if not os.path.exists(path):
        return
    try:
        import stat
        os.chmod(path, stat.S_IWRITE)
    except:
        pass

    for i in range(retries):
        try:
            shutil.rmtree(path, onerror=on_rm_error)
            return
        except OSError as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                if os.name == 'nt':
                    try:
                        os.system(f'rmdir /S /Q "{path}"')
                    except Exception:
                        pass

# --- Core Processor ---

class CASMEParallelLoader:
    def __init__(self, excel_path, local_root="./data/casme_raw", processed_root="./data/casme_processed", max_workers=5):
        self.host = os.getenv("FTP_HOST")
        self.user = os.getenv("FTP_USER")
        self.password = os.getenv("FTP_PASS")
        
        if not all([self.host, self.user, self.password]):
            raise ValueError("Missing FTP credentials in .env file (FTP_HOST, FTP_USER, FTP_PASS)")
            
        self.excel_path = excel_path
        self.local_root = local_root
        self.processed_root = processed_root
        self.max_workers = max_workers
        
        # Configure FTP paths per dataset structure
        self.parts_config = {
            "Part_A": "part_A/data/Compressed_version1_seperate_compress",
            "Part_B": "part_B/Compressed_version1_seperate_compress", 
        }
        
        self.annotations = {} 
        self._load_annotations()

    def _load_annotations(self):
        if not os.path.exists(self.excel_path):
            print(f"Error: Annotation file not found at {self.excel_path}")
            return

        try:
            df = pd.read_excel(self.excel_path)
            col_map = {c.lower(): c for c in df.columns}
            
            # Subject Column
            sub_col = col_map.get('subject') or col_map.get('sub')
            if not sub_col:
                 matches = [c for c in col_map.values() if 'sub' in c.lower()]
                 if matches: sub_col = matches[0]

            # Video/Sequence Column
            vid_col = col_map.get('seq') or col_map.get('filename') or col_map.get('video')
            if not vid_col:
                 matches = [c for c in col_map.values() if 'seq' in c.lower() or 'file' in c.lower() or 'video' in c.lower()]
                 if matches: vid_col = matches[0]

            # Onset / Apex Columns
            onset_col = col_map.get('onset') or [c for c in col_map.values() if 'onset' in c.lower()][0]
            apex_col = col_map.get('apex') or [c for c in col_map.values() if 'apex' in c.lower()][0]

            # Emotion Column (optional)
            emo_col = col_map.get('emotion') or col_map.get('est_emotion') or col_map.get('label')
            if not emo_col:
                 matches = [c for c in col_map.values() if 'emo' in c.lower() or 'label' in c.lower()]
                 if matches: emo_col = matches[0]

            print(f"Using Query Columns: Subject='{sub_col}', Video='{vid_col}', Onset='{onset_col}', Apex='{apex_col}'")
            
            for idx, row in df.iterrows():
                try:
                    sub_raw = str(row[sub_col])
                    match = re.search(r'(\d+)', sub_raw)
                    if match:
                        sub_id = str(int(match.group(1))) # Normalize '01' to '1'
                        
                        if sub_id not in self.annotations:
                            self.annotations[sub_id] = []
                        
                        if pd.notna(row[onset_col]) and pd.notna(row[apex_col]):
                            record = {
                                'video': str(row[vid_col]).strip(),
                                'onset': int(row[onset_col]),
                                'apex': int(row[apex_col])
                            }
                            if emo_col and pd.notna(row[emo_col]):
                                record['emotion'] = str(row[emo_col]).strip()
                                
                            self.annotations[sub_id].append(record)
                except Exception:
                    continue 
            
            print(f"Loaded annotations for {len(self.annotations)} subjects.")

        except Exception as e:
            print(f"Failed to load annotations: {e}")

    def _get_ftp_connection(self):
        """Creates a new FTP connection (one per thread required)."""
        ftp = ftplib.FTP(self.host)
        ftp.login(self.user, self.password)
        return ftp

    def scan_remote_files(self):
        """Scans FTP to find all valid subject zip files needed."""
        print("Scanning FTP server for subject files...")
        tasks = []
        
        with self._get_ftp_connection() as ftp:
            for part_name, relative_path in self.parts_config.items():
                try:
                    file_list = []
                    ftp.retrlines(f'NLST {relative_path}', file_list.append)
                    
                    for fname in file_list:
                        if not fname.lower().endswith('.zip'): continue
                        
                        # Check if this subject is in our Excel file
                        base_name = os.path.basename(fname.replace('\\', '/'))
                        match = re.search(r'(\d+)', base_name)
                        if not match: continue
                        
                        sub_id = str(int(match.group(1)))
                        if sub_id in self.annotations:
                            # Construct remote path
                            if '/' in fname or '\\' in fname:
                                remote_path = fname.replace('\\', '/')
                            else:
                                remote_path = f"{relative_path}/{fname}"
                            remote_path = remote_path.replace('//', '/')
                            
                            tasks.append({
                                'sub_id': sub_id,
                                'part': part_name,
                                'remote_path': remote_path,
                                'filename': base_name
                            })
                except ftplib.error_perm as e:
                    print(f"Skipping {part_name}: {e}")
        
        # Deduplication
        unique_tasks = []
        seen = set()
        for t in tasks:
            key = (t['sub_id'], t['part'])
            if key not in seen:
                seen.add(key)
                unique_tasks.append(t)

        print(f"Found {len(unique_tasks)} unique subject archives to process.")
        
        # --- Filter out already processed subjects ---
        final_tasks = []
        skipped_count = 0
        for t in unique_tasks:
            sub_dir = os.path.join(self.processed_root, f"sub{t['sub_id']}")
            # Check if directory exists and is not empty (contains at least one processed clip)
            if os.path.exists(sub_dir) and any(os.scandir(sub_dir)):
                skipped_count += 1
            else:
                final_tasks.append(t)
                
        if skipped_count > 0:
            print(f"Skipping {skipped_count} subjects that are already processed.")
            print(f"Remaining tasks: {len(final_tasks)}")
            
        return final_tasks

    def process_single_subject(self, task_info, position_id):
        """
        Worker function: Download -> Extract -> Filter -> Save -> Cleanup
        """
        sub_id = task_info['sub_id']
        remote_path = task_info['remote_path']
        fname = task_info['filename']
        
        # Setup specific progress bar for this thread
        # Position 1+ to leave 0 for the main bar
        # leave=False ensures the bar disappears/clears when done, so it can be reused
        desc_text = f"Sub {sub_id}"
        t_pbar = tqdm(total=0, position=position_id, desc=desc_text, leave=False,
                      unit='B', unit_scale=True, unit_divisor=1024,
                      miniters=1, smoothing=0.1,
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')
        
        t_pbar.set_description(f"Sub-{sub_id}: Connect")
        
        # Use a unique local dir for this thread/task to avoid collision
        safe_name = f"sub{sub_id}_{task_info['part']}"
        local_zip = os.path.join(self.local_root, f"{safe_name}.zip")
        extract_dir = os.path.join(self.local_root, safe_name)
        
        try:
            # 1. DOWNLOAD
            ftp = self._get_ftp_connection()
            try:
                # Get file size
                size = 0
                try: size = ftp.size(remote_path)
                except: pass
                
                t_pbar.reset(total=size)
                t_pbar.set_description(f"Sub-{sub_id}: DL")
                
                os.makedirs(self.local_root, exist_ok=True)
                with open(local_zip, 'wb') as f:
                    def callback(data):
                        f.write(data)
                        t_pbar.update(len(data))
                    ftp.retrbinary(f"RETR {remote_path}", callback)
            finally:
                ftp.quit()

            if os.path.getsize(local_zip) < 100:
                t_pbar.close()
                return f"[Sub {sub_id}] Failed: Zip file too small or empty."

            # 2. EXTRACT
            t_pbar.set_description(f"Sub-{sub_id}: Unzip")
            robust_rmtree(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)
            try:
                with zipfile.ZipFile(local_zip, 'r') as zf:
                    zf.extractall(extract_dir)
            except zipfile.BadZipFile:
                t_pbar.close()
                return f"[Sub {sub_id}] Failed: Bad Zip File."
            finally:
                try: os.remove(local_zip)
                except: pass

            # 3. LOCATE VIDEO FOLDERS & EXTRACT FRAMES
            t_pbar.set_description(f"Sub-{sub_id}: Proc")
            processed_count = 0
            out_sub_dir = os.path.join(self.processed_root, f"sub{sub_id}")
            os.makedirs(out_sub_dir, exist_ok=True)
            
            # Create a clean folder map to avoid walking many times
            folder_map = {}
            for root, dirs, files in os.walk(extract_dir):
                for d in dirs:
                    folder_map[d.lower()] = os.path.join(root, d)

            for vid_info in self.annotations[sub_id]:
                vid_name = vid_info['video']
                onset = vid_info['onset']
                apex = vid_info['apex']

                # Strategy:
                # 1. Find folder matching vid_name
                # 2. Look for 'color' subfolder inside it (structure: spNO.10\a\color\*.jpg)
                
                v_target = vid_name.lower()
                target_path = None
                
                if v_target in folder_map:
                    target_path = folder_map[v_target]
                else:
                    # Fallback partial match search
                    for k, v in folder_map.items():
                        if k.endswith(f"_{v_target}") or k.endswith(f" {v_target}"):
                            target_path = v
                            break

                # Ensure we point to the color folder if present
                if target_path:
                    # Check for 'color' subdir
                    potential_color = os.path.join(target_path, "color")
                    if os.path.exists(potential_color):
                        target_path = potential_color
                    # Fallback: check if 'color' is capitalized or something, though Windows is case-insensitive usually
                    elif os.path.exists(os.path.join(target_path, "Color")):
                        target_path = os.path.join(target_path, "Color")
                
                if not target_path: 
                    continue

                if not os.path.exists(target_path): 
                    continue
                
                # Get files for frame matching
                try: all_files = sorted(os.listdir(target_path))
                except: continue

                def find_frame(num):
                    # Direct match attempts
                    priors = [f"img{num}.jpg", f"reg_img{num}.jpg", f"{num}.jpg", f"image_{num}.jpg"]
                    for p in priors:
                        full_p = os.path.join(target_path, p)
                        if os.path.exists(full_p): return full_p
                    
                    # Regex search if direct match fails
                    for f in all_files:
                        if not f.lower().endswith(('.jpg', '.png')): continue
                        nums = re.findall(r'\d+', f)
                        if nums and int(nums[-1]) == num:
                            return os.path.join(target_path, f)
                    return None

                p_onset = find_frame(onset)
                p_apex = find_frame(apex)

                if p_onset and p_apex:
                    clip_name = f"{vid_name}_{onset}_{apex}"
                    clip_path = os.path.join(out_sub_dir, clip_name)
                    os.makedirs(clip_path, exist_ok=True)
                    shutil.copy2(p_onset, os.path.join(clip_path, "onset.jpg"))
                    shutil.copy2(p_apex, os.path.join(clip_path, "apex.jpg"))
                    
                    with open(os.path.join(clip_path, "info.txt"), "w") as f:
                        f.write(f"Subject: {sub_id}\nVideo: {vid_name}\nOnset: {onset}\nApex: {apex}\n")
                        if 'emotion' in vid_info:
                            f.write(f"Emotion: {vid_info['emotion']}\n")
                    processed_count += 1

            # 4. CLEANUP
            t_pbar.set_description(f"Sub-{sub_id}: Clean")
            robust_rmtree(extract_dir)
            
            if processed_count > 0:
                t_pbar.close()
                return f"[Sub {sub_id}] Success: Processed {processed_count} videos."
            else:
                t_pbar.close()
                return f"[Sub {sub_id}] Warning: Downloaded ok, but no matching videos/frames found."

        except Exception as e:
            t_pbar.close()
            return f"[Sub {sub_id}] Error: {str(e)}"

    def run(self):
        tasks = self.scan_remote_files()
        if not tasks:
            print("No tasks found.")
            return

        print(f"Starting parallel processing with {self.max_workers} threads...")
        
        # Position management queue: contains [1, 2, ... max_workers]
        # Position 0 is reserved for the main bar
        pos_queue = queue.Queue()
        for i in range(1, self.max_workers + 1):
            pos_queue.put(i)

        # Use efficient tqdm settings for multi-threading
        tqdm_args = {
            "total": len(tasks),
            "desc": "Total Progress",
            "position": 0,
            "leave": True,
            "dynamic_ncols": True,
            "unit": "sub",
            "smoothing": 0.1,  # More responsive speed updates
            "bar_format": "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            
            # Helper to manage position assignment for each thread
            def wrapped_task(task):
                 pos = pos_queue.get()
                 try:
                     return self.process_single_subject(task, pos)
                 finally:
                     pos_queue.put(pos)

            future_to_sub = {executor.submit(wrapped_task, t): t['sub_id'] for t in tasks}
            
            # Main Progress bar at position 0
            with tqdm(**tqdm_args) as pbar:
                for future in as_completed(future_to_sub):
                    sub_id = future_to_sub[future]
                    try:
                        result = future.result()
                        # Use pbar.write to print without breaking the progress bar layout
                        if result:
                            pbar.write(result)
                    except Exception as exc:
                        pbar.write(f"[Sub {sub_id}] Exception: {exc}")
                    pbar.update(1)

        print("\nAll processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASME3 Dataset Downloader and Preprocessing")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel download threads")
    parser.add_argument("--local_root", type=str, default="./data/casme_raw", help="Temp folder for zip downloads")
    parser.add_argument("--processed_root", type=str, default="./data/casme_processed", help="Output folder for extracted frames")
    parser.add_argument("--excel", type=str, default=r"data\casme\cas(me)3_part_A_MaE_label_JpgIndex_v2_emotion.xlsx", help="Path to annotation Excel file")
    
    args = parser.parse_args()
    
    print(f"Initializing CASME Loader with:")
    print(f"  Excel: {args.excel}")
    print(f"  Temp Dir: {args.local_root}")
    print(f"  Output Dir: {args.processed_root}")
    print(f"  Workers: {args.workers}")
    
    loader = CASMEParallelLoader(args.excel, local_root=args.local_root, processed_root=args.processed_root, max_workers=args.workers)
    loader.run()
