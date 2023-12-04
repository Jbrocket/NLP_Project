import re





if __name__ == "__main__":
    infilename = "reddit_dirty_dev.txt"
    outfilename = "reddit_clean_dev.txt"

    swear_list = ["fuck", "shit", "bitch", "dick", "cunt", "wanker", "pussy", "damn", "ass", "asshole", "bastard", "dickhead", "goddamn"]
    pattern = swear_list.pop(0)
    pattern = pattern + " |" + pattern 
    for swear in swear_list:
        pattern += "|" + swear + " |" + swear

    # print(pattern)

    infile = open(infilename, "r")
    outfile = open(outfilename, "w")

    for line in infile:
        cur_string = ""
        for char in line:
            if char == "”" or char == '"' or char == "“" or char == ",":
                continue
            else:
                cur_string += char
        # cur_string = " ".join(cur_string.split())
        # cur_string = cur_string.replace(u'\xa0')
        outfile.write(re.sub(pattern, '', cur_string, flags=re.IGNORECASE))

    infile.close()
    outfile.close()