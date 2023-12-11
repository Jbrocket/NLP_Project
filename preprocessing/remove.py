import re

# Read the file with sentences containing swear words
input_file_path = 'reddit_data/src/dirty.txt'

TARGETED_WORDS = ["fuck", "shit", "bitch", "dick", "cunt", "wanker", "pussy", "damn", "ass", "asshole", "bastard", "dickhead", "goddamn", "mufucka", "retard"]

with open(input_file_path, 'r', encoding='utf-8') as input_file:
    sentences_with_swear_words = input_file.readlines()

# Initialize a list to store sentences without swear words
cleaned_sentences = []

for sentence in sentences_with_swear_words:
    words = sentence.split()
    cleaned_sentence = ' '.join(word for word in words if not any(swear_word.lower() in word.lower() for swear_word in TARGETED_WORDS))
    cleaned_sentence = cleaned_sentence.strip()
    if cleaned_sentence:
        cleaned_sentences.append(cleaned_sentence)
    else:
        cleaned_sentences.append("Dang")

# Save cleaned sentences to a new file
output_file_path = 'cleaned_sentences.txt'

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for sentence in cleaned_sentences:
        output_file.write(sentence + '\n')