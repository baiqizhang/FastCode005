# running image clustering job
wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.11/opencv-2.4.11.zip?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fopencvlibrary%2Ffiles%2Fopencv-unix%2F2.4.11%2F&ts=1459639887&use_mirror=nbtelecom
unzip opencv-2.4.11.zip\?r\=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fopencvlibrary%2Ffiles%2Fopencv-unix%2F2.4.11%2F

sudo yum install cmake
mkdir build
cd build/
cmake ..
sudo pip install pyinstaller
vim ~/.bash_profile
export PYTHONPATH=$PYTHONPATH:/home/hadoop/opencv-2.4.11/build/lib

# pyinstaller 
python make_feature.py https://s3-us-west-2.amazonaws.com/fastcode.team/oxbuild_selected/training/all_souls_000040.jpg

pyinstaller --onefile make_feature.py
cp dist/make_feature make_feature
python mapper.py  < input.txt

# hadoop commands
hadoop fs -put input.txt
hadoop-streaming -mapper mapper.py -reducer cat -file make_feature -file mapper.py -input classification.txt -output final_output
hadoop fs -cat output0/*

#
find *.py | awk '{print "s3://bucket/folder/"$0}' > input.txt



# =======================================================================
# running image classification job

hadoop-streaming -mapper mapper.sh -reducer reducer.sh -file eval_oxbuild.lua -file mapper.sh -file reducer.sh -file oxbuild.dat -input input.txt -output sortout