import pyconll


def read_conllu(path):
    data = pyconll.load_from_file(path)
    tagged_sentences = list()
    t = 0
    for sentence in data:
        tagged_sentence = list()
        for token in sentence:
            if token.upos and token.form:
                t += 1
                tagged_sentence.append((token.form.lower(), token.upos))
        tagged_sentences.append(tagged_sentence)
    return tagged_sentences


def tag_sequence(sentences):
    return [[t for w, t in sentence] for sentence in sentences]


def text_sequence(sentences):
    return [[w for w, t in sentence] for sentence in sentences]


def sentence_split(sentences, max_len):
    new_sentence = list()
    for data in sentences:
        new_sentence.append(([data[x:x + max_len] for x in range(0, len(data), max_len)]))
    new_sentence = [val for sublist in new_sentence for val in sublist]
    return new_sentence


def convert_ner_format(text, label, file):
    with open(file, 'w') as f:
        words = 0
        i = 0
        for zip_i in zip(text, label):
            a, b = tuple(zip_i)
            for r in range(len(a)):
                item = a[r]+' '+b[r]
                f.write("%s\n" % item)
                words += 1
            f.write("\n")
            i += 1
            #if i==3: break
    print('Sentences:', i, 'Words:', words)
