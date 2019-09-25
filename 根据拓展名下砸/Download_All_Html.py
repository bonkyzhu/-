import re

type_name = input("输入你所需要下载的文件名后缀:\n")

html = open("html.txt").read()

pattern = re.compile("https?://.+?"+type_name)

# pattern = re.compile("- Paper -.+\[.+\]\((.+)\)")

# - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
# - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
matches = re.findall(pattern, html)

for match in matches:
    print(match)

# for match in matches: 
#     print("http://ai.berkeley.edu/"+match)
