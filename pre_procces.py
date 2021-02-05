import os
import spacy
import utils
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, \
    strip_numeric, remove_stopwords, strip_short

nlp = spacy.load("en_core_web_sm")

COSTUM_WORDS = ['page', 'interpretation', 'question', 'session', 'honours', 'honour', 'redacted', 'redact', 'kwon', 'molto',
                'registrar', 'liu', 'parker', 'mumba', 'mccloskey', 'karibi', 'piletta', 'zanin', 'hunt', 'delvoie', 'defence',
                'korner', 'stewart', 'robinson', 'karnavas', 'kay', 'chamber', 'edgerton', 'riad', 'rule', 'bis', 'jones',
                'hvo', 'smith', 'strugar', 'guy', 'vasiljevic', 'scott', 'kehoe', 'whyte', 'clark', 'jorda', 'kilometre',
                'ryneveld', 'williams', 'harmon', 'vance', 'owen', 'butler', 'vance', 'emmerson', 'moore', 'antonetti',
                'hall', 'brown', 'article', 'statute', 'rodrigues', 'agius', 'constitution', 'orie', 'lattanzi', 'protocol',
                'tribunal', 'law', 'exhibit', 'witness', 'protocol', 'agreement', 'rules', 'private', 'morrissey', 'status'
                'potocari', 'ackerman', 'picard', 'moskowitz', 'icty', 'penal', 'amendment', 'groome', 'evidence', 'bos',
                'court', 'crisis', 'staff', 'main', 'nice', 'assembly', 'blank', 'whereupon', 'nicholls', 'interior', 'harhoff',
                'tieger', 'appeals', 'recess', 'sutherland', 'thayer', 'saxon', 'niemann', 'prosecutor', 'interpreter',
                'weiner', 'metre', 'hayman', 'marcus', 'marcussen', 'wubben', 'main', 'board', 'koumjian', 'hoepfel',
                'bench', 'bbc', 'shall']


def filter_page_nums_and_redacted():
    for root, dirs, files in os.walk("ICTYTextFiles", topdown=False):
        for name in files:
            with open(os.path.join(root, name), 'r') as f, open(f"FilteredReadableText/{name}", 'w') as fout:
                text = f.readlines()
                lines = []
                for line in text:
                    if '(redacted)' in line:
                        continue
                    line = line.lstrip('0123456789.- ')
                    lines.append(line)
                fout.writelines(lines)


def lemmatize(file):
    with open(file, 'r') as f_in:
        doc = nlp(f_in.read())
        with open(f"WordsLemantize/{os.path.basename(file)}", 'w') as f_out:
            for word in doc:
                f_out.write(f'{word.lemma_} ')
        del doc


def clean(file):
    with open(file, 'r') as f_in:
        clened_words = preprocess_string(f_in.read(),
                                         filters=[strip_tags, strip_punctuation, strip_multiple_whitespaces,
                                                  strip_numeric, remove_stopwords, strip_short])
        with open(f"CleanedText/{os.path.basename(file)}", 'w') as f_out:
            for word in clened_words:
                if word == 'PRON': continue
                f_out.write(f'{word} ')


def clean_word(word):
    word = word.lower()
    if word in COSTUM_WORDS: return ''
    elif word == 'serbs' : return 'serb'
    elif word == 'serbian' : return 'serb'
    elif word == 'muslims' : return 'muslim'
    else: return word



def costum_clean(file):
    with open(file, 'r') as f_in:
        words = f_in.read().split()
        with open(f"CleanedText_Costum/{os.path.basename(file)}", 'w') as f_out:
            for word in words:
                f_out.write(f'{clean_word(word)} ')


# sizes = []
# for root, dirs, files in os.walk("FilteredText", topdown=False):
#     for name in files:
#         sizes.append(int(os.stat(os.path.join(root, name)).st_size/ 1000))
#
# with open('file_sizes.csv', 'w') as f:
#     for s in sizes:
#         f.write(f'{s}\n')

# print('Done')


if __name__ == '__main__':
    # utils.run_parrlell_on_dir("FilteredReadableText", lemmatize)
    # utils.run_parrlell_on_dir("WordsLemantize", clean)
    utils.run_parrlell_on_dir("CleanedText_nopron", costum_clean)
    print("Done")
