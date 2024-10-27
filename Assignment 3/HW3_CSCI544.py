import pandas as pd
import json
from sklearn.metrics import classification_report

vocab_cols = ["word", "index", "count"]
threshold = 3

train_path = "data/train"
dev_path = "data/dev"
test_path = "data/test"


def getWordTagPairWithStartEnd(s):
    s = s.split("\n")
    sentence = [("<START>", "START")]
    for line in s:
        _, word, tag = line.split("\t")
        word = word.lower()
        sentence.append((word, tag))
    sentence.append(("<END>", "END"))

    return sentence


def readSentencesWithStartEnd(s):
    with open(s, "r") as fin:
        sentences = "".join(list(fin)).split("\n\n")
        sentences[-1] = sentences[-1].rstrip()

        sentences = list(
            map(getWordTagPairWithStartEnd, sentences))

        return sentences


def getWordWithoutTag(s):
    s = s.split("\n")
    sentence = []
    for line in s:
        _, word, _ = line.split("\t")
        word = word.lower()
        sentence.append(word)

    return sentence


def getWordTagPair(s):
    s = s.split("\n")
    sentence = []
    for line in s:
        _, word, tag = line.split("\t")
        word = word.lower()
        sentence.append((word, tag))

    return sentence


def readSentencesWithTagWithoutStartEnd(s):
    with open(s, "r") as fin:
        sentences = "".join(list(fin)).split("\n\n")
        sentences[-1] = sentences[-1].rstrip()

        sentences = list(map(getWordTagPair, sentences))

        return sentences


def readSentencesWithoutTag(s):
    with open(s, "r") as fin:
        sentences = "".join(list(fin)).split("\n\n")
        sentences[-1] = sentences[-1].rstrip()

        sentences = list(map(getWordWithoutTag, sentences))

        return sentences


def getTestWords(s):
    s = s.split("\n")
    sentence = []
    for line in s:
        _, word = line.split("\t")
        word = word.lower()
        sentence.append(word)

    return sentence


def readTestSentences(s):
    with open(s, "r") as fin:
        sentences = "".join(list(fin)).split("\n\n")
        sentences[-1] = sentences[-1].rstrip()

        sentences = list(map(getTestWords, sentences))

        return sentences


def createVocab(sentences):
    vocabFreq = dict()

    for sentence in sentences:
        for word, _ in sentence:
            if word in vocabFreq:
                vocabFreq[word] += 1
            else:
                vocabFreq[word] = 1

    count = 0
    updatedVocabFreq = dict()
    for key in vocabFreq:
        if vocabFreq[key] < threshold:
            count += vocabFreq[key]
        else:
            updatedVocabFreq[key] = vocabFreq[key]

    vocab_csv = []
    for key in updatedVocabFreq:
        vocab_csv.append([key, updatedVocabFreq[key]])
    vocab_csv.sort(key=lambda x: x[1], reverse=True)
    vocab_csv.insert(0, ["<unk>", count])

    vocab_df = pd.DataFrame(vocab_csv)
    vocab_df = vocab_df.rename(columns={0: "word", 1: "count"})
    vocab_df["index"] = vocab_df.index + 1
    vocab_df = vocab_df[vocab_cols]
    vocab_df.to_csv("vocab.txt", sep="\t")

    updatedVocabFreq["<unk>"] = count

    s = 0
    for key, value in updatedVocabFreq.items():
        s += value

    print("Number of words: ", len(updatedVocabFreq))
    print("Total Freq: ", s)
    print("Freq of <unk>: ", updatedVocabFreq["<unk>"])

    return updatedVocabFreq


def updateSentenceWithUnknown(sentence, vocab):
    s = []
    for word, tag in sentence:
        if word in vocab:
            s.append((word, tag))
        else:
            s.append(("<unk>", tag))
    return s


def updateAllSentencesWithUnknown(sentences, vocab):
    sentences = list(
        map(
            lambda sentence: updateSentenceWithUnknown(sentence, vocab),
            sentences
        )
    )
    return sentences


def generateTagFrequencies(sentences):
    tagsFreq = dict()
    for sentence in sentences:
        for _, tag in sentence:
            if tag in tagsFreq:
                tagsFreq[tag] += 1
            else:
                tagsFreq[tag] = 1
    return tagsFreq


def createHMMModel(tagsFreq, sentences):
    transition_parameter = dict()
    emission_parameter = dict()

    for sentence in sentences:
        for i in range(len(sentence) - 1):
            _, s = sentence[i]
            _, s_dash = sentence[i+1]

            if (s, s_dash) in transition_parameter:
                transition_parameter[(s, s_dash)] += 1
            else:
                transition_parameter[(s, s_dash)] = 1

        for i in range(len(sentence)):
            x, s = sentence[i]

            if (s, x) in emission_parameter:
                emission_parameter[(s, x)] += 1
            else:
                emission_parameter[(s, x)] = 1

    for key in transition_parameter:
        s = key[0]
        count_s = tagsFreq[s]
        transition_parameter[key] = transition_parameter[key] / count_s

    for key in emission_parameter:
        s = key[0]
        count_s = tagsFreq[s]
        emission_parameter[key] = emission_parameter[key] / count_s

    model = {
        "transition": {str(key): value for key, value in transition_parameter.items()},
        "emission": {str(key): value for key, value in emission_parameter.items()}
    }

    print("Transition Parameterss", len(transition_parameter))
    print("Emission parameters", len(emission_parameter))

    with open('hmm.json', 'w') as f:
        json.dump(model, f, indent=2)

    model = {
        "transition": transition_parameter,
        "emission": emission_parameter
    }

    return model


def greedyDecoding(model, sentences, tagsFreq):
    transition_parameter = model["transition"]
    emission_parameter = model["emission"]

    predicted_sentences = []

    for sentence in sentences:
        prev_tag = "START"
        sentence_prediction = []

        for word in sentence:
            max_p = 0
            word_tag = ""

            for tag in tagsFreq:
                t = (prev_tag, tag)
                e = (tag, word)

                if t in transition_parameter and e in emission_parameter:
                    p = transition_parameter[t] * emission_parameter[e]
                    if (p > max_p):
                        max_p = p
                        word_tag = tag

            if max_p == 0:
                word_tag = "NN"

            sentence_prediction.append((word, word_tag))
            prev_tag = word_tag

        predicted_sentences.append(sentence_prediction)

    return predicted_sentences

def getDevAccuracy(file, sentences):
    actual_sentences = readSentencesWithTagWithoutStartEnd(file)

    actual_tags = []
    for sentence in actual_sentences:
        for _, tag in sentence:
            actual_tags.append(tag)

    predicted_tags = []
    for sentence in sentences:
        for _, tag in sentence:
            predicted_tags.append(tag)

    report = classification_report(
        actual_tags, predicted_tags, output_dict=True, zero_division=0
    )
    accuracy = report["accuracy"]

    print(accuracy)


def writeTestOutput(sentences, fileName):
    with open(fileName, "w") as fout:
        for sentence in sentences:
            for index, (word, tag) in enumerate(sentence):
                s = str(index + 1) + "\t" + word + "\t" + tag + "\n"
                fout.write(s)
            fout.write("\n")


train_sentences = readSentencesWithStartEnd(train_path)
vocab = createVocab(train_sentences)
train_sentences = updateAllSentencesWithUnknown(train_sentences, vocab)
tagsFreq = generateTagFrequencies(train_sentences)
model = createHMMModel(tagsFreq, train_sentences)

dev_sentences = readSentencesWithoutTag(dev_path)
greedy_dev_predicted_sentences = greedyDecoding(model, dev_sentences, tagsFreq)
print("Greedy Decoding Accuracy")
getDevAccuracy(dev_path, greedy_dev_predicted_sentences)
print()

test_sentences = readTestSentences(test_path)
greedy_test_predicted_sentences = greedyDecoding(
    model, test_sentences, tagsFreq)
writeTestOutput(greedy_test_predicted_sentences, "greedy.out")
