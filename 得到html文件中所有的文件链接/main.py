import re
import os
from multiprocessing.pool import ThreadPool

file = 'web.html'
type = 'gz'
link = 'https?://.+{}'.format(type)
destination = 'download'

content = open(file).read()

pattern = re.compile(link)
match = re.findall(pattern, content)


def download(m):
  os.system(f"wget -P {destination} {m}")

print(f"以下是所有匹配到的{len(match)}条链接")

for m in match:
  print(m)

judge = input("是否下载：(y/n)\t")
if judge == 'y':
  pool = ThreadPool(processes=20)  
  pool.map(download, match)
  pool.close() 