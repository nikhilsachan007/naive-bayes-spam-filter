import os
import math
import email
import re


def smooth(words, smoothing):
    denom = 0
    for t in words:
        denom += words[t]
    denom += smoothing*(len(words) + 1)
    ret = {}
    for t in words:
        ret[t] = math.log((float)(words[t] + smoothing)/denom)
    ret["<UNK>"] = math.log((float)(smoothing)/denom)

    return ret


def load_tokens(email_path):
    ret = []
    with open(email_path) as f:
        message = email.message_from_file(f)
        for line in email.iterators.body_line_iterator(message):
            lst = line.split()
            ret += lst

    return ret


def load_subject_tokens(email_path):
    ret = []
    with open(email_path) as f:
        message = email.message_from_file(f)
        ret = message.get('From').split()

    return ret


def load_sender_tokens(email_path):
    ret = []
    with open(email_path) as f:
        message = email.message_from_file(f)
        ret = re.findall(r"[\w\d]+", message.get('From'))

    return ret


def log_probs(tokens, smoothing):
    words = {}

    for t in tokens:
        if t in words:
            words[t] += 1
        else:
            words[t] = 1
    return smooth(words, smoothing)


def log_bigram_probs(email_paths, smoothing):
    words = {}

    for path in email_paths:
        tokens = load_tokens(path)
        for i in xrange(len(tokens)):
            if i == len(tokens) - 1:
                word = tokens[i-1] + tokens[i]
            else:
                word = tokens[i] + tokens[i+1]

            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    return smooth(words, smoothing)


def log_punc_probs(tokens, smoothing):
    words = {}

    for t in tokens:
        for c in t:
            if c in words:
                words[c] += 1
            else:
                words[c] = 1
    return smooth(words, smoothing)


def log_caps_probs(tokens, smoothing):
    words = {True: 0, False: 0}

    for t in tokens:
        words[t.isupper()] += 1

    denom = 0
    for t in words:
        denom += words[t]
    ret = {}
    for t in words:
        ret[t] = math.log((float)(words[t])/denom)

    return ret


class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir):
        spam_files = [os.path.join(spam_dir, f) for f in os.listdir(spam_dir)]
        ham_files = [os.path.join(ham_dir, f) for f in os.listdir(ham_dir)]

        spam_tokens = [t for path in spam_files for t in load_tokens(path)]
        ham_tokens = [t for path in ham_files for t in load_tokens(path)]

        smoothing = 1e-5
        self.spam = log_probs(spam_tokens, smoothing)
        self.ham = log_probs(ham_tokens, smoothing)

        smoothing = 1e-36
        self.b_spam = log_bigram_probs(spam_files, smoothing)
        self.b_ham = log_bigram_probs(ham_files, smoothing)

        smoothing = 1e-36
        self.punc_spam = log_punc_probs(spam_tokens, smoothing)
        self.punc_ham = log_punc_probs(ham_tokens, smoothing)

        self.caps_spam = log_caps_probs(spam_tokens, smoothing)
        self.caps_ham = log_caps_probs(ham_tokens, smoothing)

        spam_tokens = [t for path in spam_files for t in
                       load_subject_tokens(path)]
        ham_tokens = [t for path in ham_files for t in
                      load_subject_tokens(path)]

        smoothing = 1e-36
        self.subject_spam = log_probs(spam_tokens, smoothing)
        self.subject_ham = log_probs(ham_tokens, smoothing)

        spam_tokens = [t for path in spam_files for t in
                       load_sender_tokens(path)]
        ham_tokens = [t for path in ham_files for t in
                      load_sender_tokens(path)]
        smoothing = 1e-36
        self.sender_spam = log_probs(spam_tokens, smoothing)
        self.sender_ham = log_probs(ham_tokens, smoothing)

        total = len(spam_files) + len(ham_files)
        self.p_spam = (float)(len(spam_files)) / total
        self.p_ham = (float)(len(ham_files)) / total

    def pred_bigram(self, tokens, spam, ham):

        for i in xrange(len(tokens)):
            if i == len(tokens) - 1:
                t = tokens[i-1] + tokens[i]
            else:
                t = tokens[i] + tokens[i+1]

            if t in self.b_ham:
                ham += self.b_ham[t]

            else:
                ham += self.b_ham["<UNK>"]

            if t in self.b_spam:
                spam += self.b_spam[t]
            else:
                spam += self.b_spam["<UNK>"]

        return (spam, ham)

    def pred_punc(self, tokens, spam, ham):
        for t in tokens:
            for c in t:
                if c in self.punc_ham:
                    ham += self.punc_ham[c]
                else:
                    ham += self.punc_ham["<UNK>"]

                if c in self.punc_spam:
                    spam += self.punc_spam[c]
                else:
                    spam += self.punc_spam["<UNK>"]

        return (spam, ham)

    def pred_caps(self, tokens, spam, ham):

        for t in tokens:
            ham += self.caps_ham[t.isupper()]
            spam += self.caps_spam[t.isupper()]

        return (spam, ham)

    def pred(self, tokens, spam, ham, spam_d, ham_d):

        for t in tokens:
            if t in spam_d:
                spam += spam_d[t]
            else:
                spam += spam_d["<UNK>"]

            if t in ham_d:
                ham += ham_d[t]
            else:
                ham += ham_d["<UNK>"]

        return (spam, ham)

    def is_spam(self, email_path):

        spam = math.log(self.p_spam)
        ham = math.log(self.p_ham)
        tokens = load_tokens(email_path)
        subject_tokens = load_subject_tokens(email_path)
        sender_tokens = load_sender_tokens(email_path)

        (spam, ham) = self.pred(tokens, spam, ham, self.spam, self.ham)

        (spam, ham) = self.pred(subject_tokens, spam, ham, self.subject_spam,
                                self.subject_ham)
        (spam, ham) = self.pred(sender_tokens, spam, ham, self.sender_spam,
                                self.sender_ham)

        (spam, ham) = self.pred_bigram(tokens, spam, ham)
        (spam, ham) = self.pred_punc(tokens, spam, ham)
        (spam, ham) = self.pred_caps(tokens, spam, ham)

        return spam > ham
