import os

def get_clips(base_dir):
    clips = set()
    if not os.path.exists(base_dir):
        return set()
    for subj in os.listdir(base_dir):
        subj_path = os.path.join(base_dir, subj)
        if os.path.isdir(subj_path):
            for clip in os.listdir(subj_path):
                clip_path = os.path.join(subj_path, clip)
                if os.path.isdir(clip_path):
                    clips.add(f"{subj}/{clip}")
    return clips

c1 = get_clips('casme_processed')
c2 = get_clips('casme_tvl1_processed_amp5')

print(f"Total in casme_processed: {len(c1)}")
print(f"Total in tvl1: {len(c2)}")
print(f"Missing in tvl1: {len(c1 - c2)}")
print(f"Extra in tvl1: {len(c2 - c1)}")
if len(c1 - c2) > 0:
    print("Some missing examples:", list(c1 - c2)[:5])

def check_files(base_dir):
    missing_flow = 0
    for subj in os.listdir(base_dir):
        subj_path = os.path.join(base_dir, subj)
        if os.path.isdir(subj_path):
            for clip in os.listdir(subj_path):
                clip_path = os.path.join(subj_path, clip)
                if os.path.isdir(clip_path):
                    if not os.path.exists(os.path.join(clip_path, 'flow.npy')):
                        missing_flow += 1
    return missing_flow

m = check_files('casme_tvl1_processed_amp5')
print(f"Missing flow.npy in tvl1: {m}")
