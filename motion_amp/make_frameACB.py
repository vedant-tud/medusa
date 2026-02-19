"""
Put it into the corresponding datasets directory, e.g. `/datasets/motion_mag_data/train/train_vid_frames` for me.
Make the original frames into frameAs, frameBs, frameCs(same as frameBs here)
"""
import os
import sys
import shutil
import glob


# Choose the dir you want
target_dirs = sys.argv[1].split('+')
dirs = sorted([i for i in os.listdir('.') if i in target_dirs])[:]

image_format_name='png'

current_dir = os.getcwd()

for d in dirs:
    print('ACB-Processing on', d)
    d_path = os.path.join(current_dir, d)
    os.chdir(d_path)
    
    if not os.path.exists('frameA'):
        os.mkdir('frameA')
    if not os.path.exists('frameC'):
        os.mkdir('frameC')
        
    files = sorted([f for f in os.listdir('.') if f.endswith('.{}'.format(image_format_name))], key=lambda x: int(os.path.splitext(x)[0]))
    
    # Copy files to frameA
    for f in files:
        shutil.copy(f, 'frameA')
        shutil.copy(f, 'frameC')

    # Remove last frame from frameA
    frameA_files = sorted(os.listdir('frameA'), key=lambda x: int(os.path.splitext(x)[0]))
    if frameA_files:
        os.remove(os.path.join('frameA', frameA_files[-1]))

    # Remove first frame from frameC
    frameC_files = sorted(os.listdir('frameC'), key=lambda x: int(os.path.splitext(x)[0]))
    if frameC_files:
        os.remove(os.path.join('frameC', frameC_files[0]))

    # Rename files in frameC to shift indices
    frameC_files = sorted(os.listdir('frameC'), key=lambda x: int(os.path.splitext(x)[0]))
    for f in frameC_files:
        name, ext = os.path.splitext(f)
        new_name = '%06d' % (int(name)-1) + ext
        os.rename(os.path.join('frameC', f), os.path.join('frameC', new_name))
    
    # Copy frameC to frameB
    if os.path.exists('frameB'):
        shutil.rmtree('frameB')
    shutil.copytree('frameC', 'frameB')
    
    # Remove original images
    for f in files:
        if os.path.exists(f):
            os.remove(f)
            
    os.chdir(current_dir)
