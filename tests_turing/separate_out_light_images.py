import os 
import shutil
from shutil import copyfile
src_path = "../../intrinsics-network/dataset/output/airplane_train"
dst_path = "../../intrinsics-network/dataset/output/only_lights"

for file in os.listdir(src_path):
	if 'lights' in file:
		shutil.copy2(os.path.join(src_path,file), dst_path)