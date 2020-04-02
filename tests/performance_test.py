import  psutil
import time
import os
import shutil
import threading

# encoding: utf-8
path_prefix = '/path/to/arctern_back/tests/process_info/'
spark_cmd = "/path/to/spark-3.0.0-preview2/bin/spark-submit --master local /path/to/arctern_back/tests/spark_test.py"
pname = 'python'

def submit_task():
   print(spark_cmd)
   os.system(spark_cmd)

def write_relationship(pid_exist):
   pid_to_write_relationship=[]
   for i in pid_exist:
     pid_to_write_relationship.append(i)
   f = open (path_prefix + 'relationship.txt','a')
   while (p1.is_alive()):
    for proc in psutil.process_iter():
      if proc.name().find(pname) >= 0 and proc.pid not in pid_to_write_relationship:
        f.write('进程ID:')
        f.write(str(proc.pid))
        f.write('\n父进程ID:')
        f.write(str(proc.ppid()))
        proc_father=psutil.Process(proc.ppid())
        f.write('\n父进程名称:')
        f.write(proc_father.name())
        f.write('\n子进程ID:')
        f.write(str(proc.children()))
        f.write('\n')
        pid_to_write_relationship.append(proc.pid)


def get_pid(pname,pid_exist):
   for proc in psutil.process_iter():
    if proc.name().find(pname) >= 0 and proc.pid not in pid_exist:
      file_name = path_prefix + str(proc.pid) + '.txt'
      f = open (file_name,'a')
      f.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
      f.write('   ') 
      f.write(str(proc.memory_info()))
      f.write('    ')
      f.write(str(proc.cpu_percent()))
      f.write('\n')
      proc_father=psutil.Process(proc.ppid())
      if proc_father.name() == 'java':
        file_name = path_prefix + str(proc_father.pid) + '.txt'
        f = open (file_name,'a')
        f.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        f.write('   ') 
        f.write(str(proc_father.memory_info()))
        f.write('    ')
        f.write(str(proc_father.cpu_percent()))
        f.write('\n')

def write_process(pname,pid_exist):
   start_time = time.time()
   while p1.is_alive():
     time.sleep(1)
     get_pid(pname,pid_exist)
   end_time = time.time()
   print('totally cost',end_time-start_time)

p1=threading.Thread(target=submit_task)

if __name__=="__main__": 
  if os.path.exists(path_prefix):
     shutil.rmtree(path_prefix)
  os.makedirs(path_prefix)
  exists_python_pid=[]
  for proc in psutil.process_iter():
     if proc.name() == 'python':
       exists_python_pid.append(proc.pid)
  print(exists_python_pid)
  #p1=threading.Thread(target=submit_task)
  p2=threading.Thread(target=write_relationship,args=(exists_python_pid,))
  p3=threading.Thread(target=write_process,args=(pname,exists_python_pid,))
  p1.start()
  time.sleep(8)
  p2.start()
  p3.start()
