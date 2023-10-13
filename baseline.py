#!/usr/bin/env python3

import random, sys, string

random.seed()
if len(sys.argv) != 3:
    print("usage: baseline.py input_file test_file")
    sys.exit(0)

input = sys.argv[1]
test = sys.argv[2]



input_file = open(input, "r")
test_file = open(test, "r")

score = 0
total = 0

for index, line in enumerate(input_file):
    line = line.split()
    line.pop(random.randint(0, len(line)-1))

    if line == test_file.readline().split():
        score += 1
    total += 1


print(f"# Correct: {score}")
print(f"Total lines: {total}")
print(f"Accuracy: {score/total}")

