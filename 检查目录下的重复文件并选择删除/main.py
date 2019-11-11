import os
from multiprocessing.pool import ThreadPool
from prettytable import PrettyTable
import os
import re

path = './'

def gci(filepath, print_file=False, abspath=True):
  #遍历filepath下所有文件，包括子目录
  lists = []
  files = os.listdir(filepath)
  for fi in files:
    fi_d = os.path.join(filepath,fi)            
    if os.path.isdir(fi_d):
      lists += gci(fi_d, print_file, abspath)                  
    else:
      if abspath:
        lists.append(fi_d)
        if print_file :
          print(fi_d)
      else:
        lists.append(fi)
        if print_file :
          print(fi)
  return lists

def process(file):
  file = [ f.replace(' ', '\ ') for f in file]
  file = [ f.replace('(', '\(') for f in file]
  file = [ f.replace(')', '\)') for f in file]

  return file

def check(abs_file):
  tmp = set()
  info = {}

  for af in abs_file:
    info[af] = os.stat(af).st_size
  
  for af in abs_file:
    af_same = set()
    for tmp_af in abs_file:
      if info[af] == info[tmp_af] : 
          af_same.add(tmp_af)
    tmp.add('\t'.join(list(af_same)))

  for t in list(tmp):
    t = t.split('\t')
    if len(t) >= 2:
      print("\n有以下重复文件")
      table = PrettyTable(["序号", "文件路径"])
      for i, f in enumerate(t):
        table.add_row([i+1, f])
      print(table)
      index = input("选择你要保留的的[不进行删除请回车]: ")
      if index != '':
        index = int(index)
        delete_file = t[0:index-1] + t[index:]
        delete(delete_file)
        print("删除完成！")
      else:
        print("没有进行删除。")

def delete(delete_file):
  for df in delete_file: 
    os.system(f"rm -f {df}")

if __name__ == '__main__':
  file = gci(path)
  process(file)

  check(file)

