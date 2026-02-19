VIDEO_NAME=baby
VIDEO_FORMAT=.avi


mkdir ${VIDEO_NAME}
ffmpeg -i ${VIDEO_NAME}${VIDEO_FORMAT} -f image2 ${VIDEO_NAME}/%06d.png
python make_frameACB.py ${VIDEO_NAME}
mkdir test_dir
mv ${VIDEO_NAME} test_dir
