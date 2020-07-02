import click
import os
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import textwrap
import numpy as np
import sys
import csv
import random
import logging

# Always append project's main dir (1 level up), for easy imports
main_project_dir = os.sep.join(os.path.normpath(__file__).split(os.sep)[:-1])
sys.path.append(main_project_dir)

@click.command()
@click.argument('path_to_dataset', type=click.Path(exists=True), required=True)
@click.argument('path_to_annotations_file', type=click.Path(exists=None), required=True)
@click.option('--random_state', type=int, default=12345)
@click.option('--feedback', is_flag=True)
@click.option('--highlight', is_flag=True)
def main(path_to_dataset, path_to_annotations_file, feedback, highlight, random_state):

    random.seed(random_state)
    np.random.seed(random_state)

    data = pd.read_csv(path_to_dataset)

    # reshuffle
    data = data.sample(len(data))

    if highlight:   # skip the ones already covered
        data = data[200:]


    print('\n'.join(textwrap.wrap("You can close at any time. If you run the script again (with the same parameters), you will continue where you left off. If you quit by entering 'q' you will see your scores so far.'")))
    print()

    if os.path.exists(path_to_annotations_file):
        # Load existing annotations:
        prev_annotations = pd.read_csv(path_to_annotations_file, index_col=0, comment="#")
        n_prev_annotations = len(prev_annotations)
        print(f"Found {n_prev_annotations} previous annotations in {path_to_annotations_file}.")

        # Check if they align with the to-be-annotated sample:
        if any(data[:n_prev_annotations].index != prev_annotations.index):
            logging.error('\n'.join(textwrap.wrap(" Oops! Existing annotations do not align with the data sample. Make sure you run it with the exact same parameters and data as before (or start a new annotations file). Or ask me, as there may also be a bug.")))
            quit()

    else:
        # Create new annotations .csv file with parameters as comment in the first line.
        with open(path_to_annotations_file, 'w') as annotations:
            annotations.write(f"# params: {path_to_dataset}, {path_to_annotations_file}, {feedback}, {random_state}\n")
            csv.writer(annotations).writerow(['', 'target', 'prediction', 'comment'])
        n_prev_annotations = 0

    for j, (i, row) in enumerate(data[n_prev_annotations:].iterrows()):

        print(f"\n----- {i} (total: {j + n_prev_annotations}) -----")
        print('\n'.join(textwrap.wrap(row['context'] if not highlight else f'H: {row["highlight"]}')))
        print(f"\nQ: {row['question']}\n")

        choice = input('Was this question evoked? y/n (append * to add comment; q to quit)\n > ').lower()
        while choice not in ["y", "n", "q", "y*", "n*"]:
            choice = input(' > '
            ).lower()
        if choice == "q":
            break

        comment = ""
        if '*' in choice:
            comment = input("\nEnter a comment:\n > ")
            if comment == "q":
                break

        choice = choice.startswith('y')

        with open(path_to_annotations_file, 'a') as annotations:
            writer = csv.writer(annotations)
            writer.writerow([i, row['evoked?'], choice, comment])

        if feedback:
            correct = choice == row['evoked?']
            print("   :)   " if correct else "   :(   ")

    print("All annotations saved to", path_to_annotations_file)

    print("\nScores so far:")
    all_annotations = pd.read_csv(path_to_annotations_file, index_col=0, comment="#")
    for score in [matthews_corrcoef, f1_score, accuracy_score]:
        print(f"  {score.__name__:>20}: {score(all_annotations['target'], all_annotations['prediction']):<6.3f}")


if __name__ == "__main__":
    main()