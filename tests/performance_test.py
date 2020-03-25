import  psutil
import time
import os
import shutil
import threading

# encoding: utf-8
path_prefix = '/home/liupeng/arctern_back/tests/process_info/'
pname = 'python'

def submit_task():
   spark_cmd = "/home/liupeng/cpp/spark/spark-3.0.0-preview2/bin/spark-submit --master local /home/liupeng/arctern_back/tests/spark_test.py"
   print(spark_cmd)
   os.system(spark_cmd)

def write_relationship(pid_exist):
   f = open (path_prefix + 'relationship.txt','a')
   for proc in psutil.process_iter():
    if proc.name().find(pname) >= 0 and proc.pid not in pid_exist:
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


def get_pid(pname,pid_exist):
   for proc in psutil.process_iter():
    if proc.name().find(pname) >= 0 and proc.pid not in pid_exist:
      #file_name = path_prefix + str(proc.pid) + '.txt'
      #file_name1 = path_prefix + 'process_info.txt'
      
      f = open (file_name,'a')
      f.write(str(proc.memory_info()))
      f.write('    ')
      f.write(str(proc.cpu_percent()))
      f.write('\n')
      proc_father=psutil.Process(proc.ppid())
      if proc_father.name() == 'java':
        file_name = path_prefix + str(proc_father.pid) + '.txt'
        f = open (file_name,'a')
        f.write(str(proc_father.memory_info()))
        f.write('    ')
        f.write(str(proc_father.cpu_percent()))
        f.write('\n')

def write_process(pname,pid_exist):
   start_time = time.time()
   end_time = time.time()
   while (end_time - start_time) < 35:
     time.sleep(1)
     end_time = time.time()
     get_pid(pname,pid_exist)
   print('totally cost',end_time-start_time)

if __name__=="__main__": 
  if os.path.exists(path_prefix):
     shutil.rmtree(path_prefix)
  os.makedirs(path_prefix)
  exists_python_pid=[]
  for proc in psutil.process_iter():
     if proc.name() == 'python':
       exists_python_pid.append(proc.pid)
  print(exists_python_pid)
  p1=threading.Thread(target=submit_task)
  p2=threading.Thread(target=write_relationship,args=(exists_python_pid,))
  p3=threading.Thread(target=write_process,args=(pname,exists_python_pid,))
  p1.start()
  time.sleep(8)
  p2.start()
  p3.start()
