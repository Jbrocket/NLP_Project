import sys
import random

if __name__ == "__main__":
    file = sys.argv[1]
    
    with open(file, 'r') as f:
        lines = f.readlines()
        
        dirty_lines = []
        for line in lines:
            dirty = line
            check_length = dirty.split()
            length = len(check_length)
            
            check_length.insert(random.randint(0, length-1), "FUCK")
            check_length.append('\n')
            dirty_lines.append(" ".join(check_length))
            
    with open(f"dirty_file.txt", "w+") as wf:
        wf.writelines(dirty_lines)