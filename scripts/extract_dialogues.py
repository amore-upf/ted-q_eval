import glob
import os
import csv
import re
import sys

# Always append project's main dir (1 level up)
sys.path.append(os.sep.join(os.path.normpath(__file__).split(os.sep)[:-1]))

PATH_TO_BOOKCORPUS = '/home/u148187/datasets/bookcorpus/out_txts'   # folder containing .txt files as crawled using https://github.com/soskek/bookcorpus.git

def extract_dialogues(path, min_dialogue_spacing, min_length):
    quote_pattern = '("|“)([^("|”)]*)[^ ]("|”)'

    with open(path) as file:

        previous_end_idx = 0
        total_idx = 0

        all_dialogues = []
        current_dialogue = []

        for i, line in enumerate(file):

            # Perhaps this line starts a dialogue turn
            current_turn = []

            # Iterate through all quotations in the line
            for j, match in enumerate(re.finditer(quote_pattern, line)):
                current_start_idx = total_idx + match.start(0)

                # If the current quotation is too far away from the previous one, start a new dialogue
                if (current_start_idx - previous_end_idx) >= min_dialogue_spacing:
                    # It might be neat to check independently for INTER_TURN_DISTANCE as well,
                    # but I'm assuming turns are always delineated by newlines OR new dialogues.
                    if current_turn != []:
                        current_dialogue.append(current_turn)
                        current_turn = []
                    if len(current_dialogue) >= min_length:
                        all_dialogues.append(current_dialogue)
                    current_dialogue = []

                quote = match.group(0).strip("\"“”")

                # For constructions like '"Blablabla," he said.'
                if quote.endswith(','):
                    next_period = re.search('\.', line[match.end()+1:])      # TODO Problem: ...said Dr. Dre...
                    next_quote = re.search(quote_pattern, line[match.end()+1:])
                    if next_period is None or next_quote is None or next_period.start() < next_quote.start():
                        quote = quote[:-1] + '.'

                # Append to current turn
                current_turn.append(quote)
                previous_end_idx = total_idx + match.end()

            # If there was a turn, append it to the dialogue
            if current_turn != []:
                current_dialogue.append(current_turn)

            total_idx += len(line)

        # If there was a dialogue, append it to the dialogues
        if len(current_dialogue) >= min_length:
            all_dialogues.append(current_dialogue)

    return all_dialogues


if __name__ == '__main__':

    bookcorpus_paths = glob.glob(PATH_TO_BOOKCORPUS + '/*.txt')
    min_dialogue_spacing = 200
    min_length = 2
    out_file = 'bookcorpus_dialogues_{}_{}.csv'.format(min_dialogue_spacing, min_length)
    if os.path.exists(out_file):
        if not input('Output file already exists. Overwrite?').startswith("y"):
            quit()
    with open(out_file, 'w') as output:
        writer = csv.writer(output);
        for i, path in enumerate(bookcorpus_paths):
            print('Book', i, 'of', len(bookcorpus_paths))
            all_dialogues = extract_dialogues(path, min_dialogue_spacing, min_length)
            for dia in all_dialogues:
                writer.writerow([os.path.basename(path)[:-4]] + [' '.join(turn) for turn in dia])
