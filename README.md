# Naive Bayes Spam Filter

This was an assignment for the AI class at Penn. We were given a two sets of spam and ham emails, contained in the `train` and `dev` folders available [here](http://www.seas.upenn.edu/~cis521/Homework/homework5_data.zip). The goal was to create a Naive Bayes classifier trained on the `train` set of emails that could achieve 99% accuracy classifying the emails in the `dev` folder. This classifier achieved a 99% accuracy on the `dev` set and a accuracy of 99.4975% on an additional withheld dataset.

## Dependencies ##
Python 2.x

## Implementation ##

This spam filter is based on six different types of features. First, I used the simple bag-of-words model and counted the number of times that each word occurred in the email bodies of the spam and ham datasets. I also added bigram features where I looked at every two adjacent words in each email and recorded the number of times I saw each bigrams. I recorded the number of times I saw each character. I also recorded the number of words that were written in all caps and the number of words that weren't. Finally, I broke up the From field and the Subject field of the email message and used them as two additional bags of words.

Given these counts, I calculate the probability of a each word or bigram or an all-caps word (feature) given spam or ham. This probability is the count of that feature divided by the count of all features in its category over the set of ham or spam emails. Laplace smoothing was applied to the probabilities. The smoothing constants were also adjusted by trying out different values and evaluating the performance on the `dev` set.

Because the probabilities are in log space, when classifying an email, I add the probability given spam or ham for each feature that we see in the email. If the sum given ham is bigger than the sum given spam, it is classified for ham or vice versa.

## Use

Initialize the `SpamFilter` object using the paths to the spam and ham training data folders.
    
    sf = SpamFilter("train/spam", "train/ham")
    
The `is_spam` function will classify an email as spam or ham

    sf.is_spam("dev/spam/spam1")
