import sys
import random

swear_words = ["fuck", "shit", "bitch", "dick", "cunt", "wanker", "pussy", "damn", "ass", "asshole", "bastard", "dickhead", "goddamn"]

files = 125

ftrain = open("../data/train/train.txt", "w+")
fdev = open("../data/dev/dev.txt", "w+")
ftest = open("../data/test/test.txt", "w+")

ftrain_dirty = open("../data/train/train_random_dirty.txt", "w+")
fdev_dirty = open("../data/dev/dev_random_dirty.txt", "w+")
ftest_dirty = open("../data/test/test_random_dirty.txt", "w+")
for i in range(1, files):
    fread = open(f"../../data_temp/raw/raw_data_{i}.txt", "r")
    lines = fread.readlines()
    nums = random.sample(range(len(lines)), 420)
    
    train = []
    dev = []
    test = []
    for i in range(400):
        train.append(lines[i])
    for i in range(400, 410):
        dev.append(lines[i])
    for i in range(410, 420):
        test.append(lines[i])
        
    for line in train:
        ftrain.write(line)
        n = random.randint(1, 3)
        line.strip('\n')
        line = line.split(' ')
        for i in range(n):
            swear = random.choice(swear_words)
            line.insert(random.randint(0, len(line)-1), swear)
        # line.append('\n')
        line = " ".join(line)
        ftrain_dirty.write(line)
        
    for line in dev:
        fdev.write(line)
        n = random.randint(1, 3)
        line.strip('\n')
        line = line.split(' ')
        for i in range(n):
            swear = random.choice(swear_words)
            line.insert(random.randint(0, len(line)-1), swear)
        # line.append('\n')
        line = " ".join(line)
        fdev_dirty.write(line)
        
    for line in test:
        ftest.write(line)
        n = random.randint(1, 3)
        line.strip('\n')
        line = line.split(' ')
        for i in range(n):
            swear = random.choice(swear_words)
            line.insert(random.randint(0, len(line)-1), swear)
        # line.append('\n')
        line = " ".join(line)
        ftest_dirty.write(line)
            
        