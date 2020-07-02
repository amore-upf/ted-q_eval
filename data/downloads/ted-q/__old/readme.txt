Three files:

- annotations.csv contains the elicited questions with their answers and highlights, for three 'genres' of text: TED-talks, DISCO-SPICE dialogues, and a made-up short story.
- verification_annotations.csv contains a second round of 'question relatedness' annotations.
- verification_aggregated_per_pair.csv contains aggregated relatedness scores based on the previous file.

Below is an explanation of the various columns of annotations.csv .

NB. The file annotations.csv also has a column 'relatedness' which contains averaged aggregated relatedness scores for a given probe point according to verification_aggregated_per_pair.csv (i.e., how related, on average, the pairs of questions evoked at that point are).

The original source texts are included separately in the folders SPICE-CR-XML and Ted-MDB-Annotations. For information about how to align our annotations with these source texts and existing annotations see further below (but I will also write some scripts).


####################################
## Columns with meta information: ##
####################################

genre: 'spoken', 'short story' (for our little training text) or 'presentation'

source: source of the excerpt/chunk to which the annotation pertains.

excerpt_number: In each experiment workers saw around 5 excerpts, numbered incrementally; with this column we can tell if the worker was closer to the start or closer to the end of the experiment.

chunk_number: Each excerpt was traversed in 'chunks' of two or three sentences; with this column we can tell if the annotation occurred at the start of an excerpt (less context) or at the end (more context).

worker: A unique made-up name for each MTurker. I haven't excluded any workers from the table, but feel free to do so in case you find a worker who's always filling in nonsense.


########################
## Annotation fields: ##
########################

type: there are four types of annotations:
- question: for every chunk workers had to provide a question;
- answer: for every chunk workers had to say if a previously evoked question was answered (on a scale 1-5). If this was >=3, they had to type the answer, highlight a passage, and the annotation type is 'answer'.
- non-answer: those cases where the answeredness judgment is <3; in these cases no answer is typed and no passage highlighted.
- evaluation: at the end of each excerpt, workers would evaluate its coherence and naturalness.

content:
- for 'question' type annotations: the answer provided by the text, in the worker's own words
- for 'answer' type annotations: the answer provided by the text, in the worker's own words

answered: 
- for 'answer'/'non-answer' type annotations: the 'answeredness' rating given
- for 'question' type annotations: the highest 'answeredness' rating for a given question

coherence: scale 1-5, for 'evaluation' type annotations, at the end of each excerpt

naturalness: scale 1-5, for 'evaluation' type annotations, at the end of each excerpt

comment: open, optional text field; for 'evaluation' type annotations, at the end of each excerpt


####################################
## Linking questions and answers: ##
####################################

potential_answers: for 'question' type annotations only: list of annotation ids (as given by first column) of all answers and non-answers pertaining to this question.

potential_question: for 'answer'/'non-answer' type annotations only: annotation id of the question to which they pertain.

best_answer: for 'question' type annotations only: the annotation id (i.e., as given by first column) of the answer with the highest 'answeredness' rating.


###############################################
## Various notions of annotation similarity: ##
###############################################

wh_type: sloppy wh-based classification of questions (with a handful of tweaks, e.g., treating 'how come' as 'why').

neighbors: list of annotation_ids of annotations of the same type, whose highlight occured on the same line. This is used for computing the various 'agreement'-type notions below.

SIF-similarity: average similarity of an annotation's content (question or answer) to each of its neighbors (previous column) according to a distributional semantic method; pretty decent.

same_wh: for 'question' type annotations only: average of whether an annotation's content (i.e., the evoked question) is of the same wh_type as its neighbors.

highlight_inclusion: whether the highlighted passage is included of that of its neighbors, on average.

highlight_overlap_wordcount: how much the highlighted passage overlaps with that of its neighbors, in number of words -- an 'absolute' measure.

highlight_overlap_IOU: how much the highlighted passage overlaps with that of its neighbors, in terms of IOU ("intersection over union"): intersection of highlights divided by union of highlights -- a more 'relative' measure.

relatedness: for 'question' type annotations only: how related a question is to its neighbours on a 0-3 scale (0 = not closely related; 3 = equivalent, according to our human judgments)

################################################################
## Info for aligning our annotations to existing annotations: ##
################################################################

TED_MDB's annotations are based on *character* number (0, 1, 2, 3, ...), whereas DISCO_SPICE's annotations are based on times (T0, T1, T2, T3, ...), where times are typically the starts and ends of *tokens*. Accordingly, depending on the source text, you will need a different set of columns for linking it to the original annotations:

For TED_MDB:
- chunk_start_char & chunk_end_char: for the start/end character positions of chunks.
- highlight_start_char & highlight_end_char: for the start/end character of highlighted passage.

For DISCO_SPICE:
- chunk_start_time & chunk_end_time: for the start/end times of chunks.
- highlight_start_time & highlight_end_time: for the start/end times of highlighted passage.

