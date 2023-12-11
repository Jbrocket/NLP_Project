PATH = "./reddit_data/"

if __name__ == "__main__":
    reddit_clean = open(f"{PATH}src/clean.txt", "r").readlines()
    reddit_dirty = open(f"{PATH}src/dirty.txt", "r").readlines()
    cleaned = open(f"{PATH}src/cleaned_sentences.txt", "r").readlines()
    tainted = open(f"{PATH}src/tainted.txt", "r").readlines()
    
    
    reddit_clean_train = open(f"{PATH}train/reddit.train.clean", "w+")
    reddit_clean_dev = open(f"{PATH}dev/reddit.dev.clean", "w+")
    reddit_clean_test = open(f"{PATH}test/reddit.test.clean", "w+")
    
    reddit_dirty_train = open(f"{PATH}train/reddit.train.dirty", "w+")
    reddit_dirty_dev = open(f"{PATH}dev/reddit.dev.dirty", "w+")
    reddit_dirty_test = open(f"{PATH}test/reddit.test.dirty", "w+")
    
    reddit_cleaned_train = open(f"{PATH}train/reddit.train.cleaned", "w+")
    reddit_cleaned_dev = open(f"{PATH}dev/reddit.dev.cleaned", "w+")
    reddit_cleaned_test = open(f"{PATH}test/reddit.test.cleaned", "w+")
    
    reddit_tainted_train = open(f"{PATH}train/reddit.train.tainted", "w+")
    reddit_tainted_dev = open(f"{PATH}dev/reddit.dev.tainted", "w+")
    reddit_tainted_test = open(f"{PATH}test/reddit.test.tainted", "w+")
    
    for i in range(8000):
        reddit_clean_train.write(reddit_clean[i])
        reddit_dirty_train.write(reddit_dirty[i])
        reddit_cleaned_train.write(cleaned[i])
        reddit_tainted_train.write(tainted[i])
        
    for i in range(8000, 8000+2500):
        reddit_clean_dev.write(reddit_clean[i])
        reddit_dirty_dev.write(reddit_dirty[i])
        reddit_cleaned_dev.write(cleaned[i])
        reddit_tainted_dev.write(tainted[i])
        
    for i in range(8000+2500, 8000+2500+3500):
        reddit_clean_test.write(reddit_clean[i])
        reddit_dirty_test.write(reddit_dirty[i])
        reddit_cleaned_test.write(cleaned[i])
        reddit_tainted_test.write(tainted[i])