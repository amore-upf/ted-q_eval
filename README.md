# Similarity or deeper understanding? Analyzing the TED-Q dataset of evoked questions

**Matthijs Westera, Jacopo Amidei and Laia Mayol**

Repository with code and data for the CoLing2020 paper:
```
@inproceedings{westera2020coling,
  title={Similarity or deeper understanding? Analyzing the TED-Q dataset of evoked questions},
  author={Matthijs Westera and Jacopo Amidei and Laia Mayol},
  booktitle = "Proceedings of the 28th International Conference on Computational Linguistics (CoLing2020)",
  year = 	 "2020",
  month = 	 "December",
  date =     "8-13",
  address =  "Barcelona, Spain",
}
```

# installing
1. create a virtual environment based on requirements.txt
2. download the Spacy model for English by (within the right environment): python -m spacy download en_core_web_lg

NB.: The BookCorpus-derived tasks are too big for github; see the data folder and the script make_tasks.py to generate them yourself.
