import requests
import sys

MAX_NUM = 3433000
cur_file = 75
cur_offset = 2202900 #broke in the middle
cur_lines = 2712
lines = set()

for i in range(MAX_NUM//100):
    ## Get data
    res = requests.get(f"https://datasets-server.huggingface.co/rows?dataset=generics_kb&config=generics_kb&split=train&offset={cur_offset}&limit=100")
    cur_offset += 100
    print(cur_offset)
    ## Parse data
    res = res.json()
    ## Put data in current file
    cur_row = 0
        
    fp = open(f"../data/raw_data_{cur_file}.txt", "a+")
    for i in range(100):
        try:
            line = res['rows'][i]['row']['generic_sentence']
        except KeyError:
            print(f"{cur_file*25000+cur_lines}")
            sys.exit(1)
        print(line)
        if line in lines:
            # sys.exit(1)
            continue
        line += "\n"
        lines.add(line)
        fp.write(line)
        cur_lines += 1

        ## Change files
        if cur_lines > 25000:
            lines = set()
            cur_file += 1
            cur_lines = 0
            fp.close()
            fp = open(f"../data/raw_data_{cur_file}.txt", "a+")
    