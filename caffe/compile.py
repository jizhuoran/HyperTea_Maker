import subprocess, os


os.chdir('/home/zrji/hypertea_maker/caffe/build')
p = subprocess.Popen(['make', '-j40'])
p.wait()
assert p.returncode == 0, "compile maker error"

os.chdir('/home/zrji/hypertea_maker/caffe/')
p = subprocess.Popen(['./build/tools/hypertea_generator'])
p.wait()
assert p.returncode == 0, "generate error"





