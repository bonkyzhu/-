import re
import os
from threadpool import ThreadPool, makeRequests

file = 'web.html'
type = 'gz'
link = 'https?://.+{}'.format(type)
destination = 'download'

content = open(file).read()

pattern = re.compile(link)
match = re.findall(pattern, content)


def download(m):
  os.system(f"wget -P {destination} {m}")

pool = ThreadPool(8)
requests = makeRequests(download,match)
map(pool.putRequest, requests)
pool.wait()