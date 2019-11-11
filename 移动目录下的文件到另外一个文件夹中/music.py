from multiprocessing.pool import ThreadPool
import os
import re

path = './'
destination = '/Users/bonky/Desktop/Music'
pattern = re.compile('.+\d.mp3') 

def gci(filepath, print_file=False, abspath=True):
  #遍历filepath下所有文件，包括子目录
  lists = []
  global pattern
  files = os.listdir(filepath)
  for fi in files:
    fi_d = os.path.join(filepath,fi)            
    if os.path.isdir(fi_d):
      lists += gci(fi_d, print_file, abspath)                  
    elif fi_d.endswith('mp3') and not re.match(pattern, fi_d):
      if abspath:
        lists.append(fi_d)
        if print_file :
          print(fi_d)
      else:
        lists.append(fi)
        if print_file :
          print(fi)
  return lists

def move(source):
  global destination
  command = f"cp -f {source} {destination}"
  os.system(command)

def process(file):
  file = [ f.replace(' ', '\ ') for f in file]
  file = [ f.replace('(', '\(') for f in file]
  file = [ f.replace(')', '\)') for f in file]

if __name__ == '__main__':
  file = gci(path, abspath=False)
  file = process(file)

  pool = ThreadPool(processes=20)  
  pool.map(move, file)
  pool.close() 