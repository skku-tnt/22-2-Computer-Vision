import torch
import glob
import unicodedata
import string
import os

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)
print(findFiles('data/names/*.txt'))
# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
# for f in findFiles('data/names/*.txt'):
#     word = f.split('\\')[1].split('.')[0]
#     # word = repr(f.split('\')[1])
#     print(word)

for filename in findFiles('data/names/*.txt'):
    category = filename.split('\\')[1].split(".")[0]
    # category = os.path.splitext(os.path.basename(filename))[0]
    # print("hello fuck..")
    # print(category)
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

print(all_categories) # show the categories..
n_categories = len(all_categories)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor