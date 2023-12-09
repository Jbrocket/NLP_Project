import sys
import random

TARGETED_WORDS = ["fuck", "shit", "bitch", "dick", "cunt", "wanker", "pussy", "damn", "ass", "asshole", "bastard", "dickhead", "goddamn", "mufucka"]

if __name__ == "__main__":
    file_input = open(sys.argv[1], "r")
    file_output = open(sys.argv[2], "w+")
    
    for line in file_input.readlines():
        n = random.randint(1, 3)
        line.strip('\n')
        line = line.split(' ')
        for i in range(n):
            swear = random.choice(TARGETED_WORDS)
            try:
                line.insert(random.randint(1, len(line)-1), swear)
            except:
                pass
        # line.append('\n')
        line = " ".join(line)
        file_output.write(line)