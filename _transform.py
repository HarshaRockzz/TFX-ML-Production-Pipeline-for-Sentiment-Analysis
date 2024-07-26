
import tensorflow as tf
import tensorflow_transform as tft

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
             "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
             "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
             "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
             "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
             "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
             "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
             "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
             "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
             "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

_LABEL_KEY = 'rating'

# Renaming transformed features
def _transformed_name(key):
    return key + '_xf'

# Define the transformations
def preprocessing_fn(inputs):

    outputs = {}

    outputs[_transformed_name('review')] = tf.strings.lower(inputs['review'])
    outputs[_transformed_name('review')] = tf.strings.regex_replace(outputs[_transformed_name('review')], r"(?:<br />)", " ")
    outputs[_transformed_name('review')] = tf.strings.regex_replace(outputs[_transformed_name('review')], "n\'t", " not ")
    outputs[_transformed_name('review')] = tf.strings.regex_replace(outputs[_transformed_name('review')], r"(?:\'ll |\'re |\'d |\'ve)", " ")
    outputs[_transformed_name('review')] = tf.strings.regex_replace(outputs[_transformed_name('review')], r"\W+", " ")
    outputs[_transformed_name('review')] = tf.strings.regex_replace(outputs[_transformed_name('review')], r"\d+", " ")
    outputs[_transformed_name('review')] = tf.strings.regex_replace(outputs[_transformed_name('review')], r"\b[a-zA-Z]\b", " ")
    outputs[_transformed_name('review')] = tf.strings.regex_replace(outputs[_transformed_name('review')], r'\b(' + r'|'.join(stopwords) + r')\b\s*', " ")

    outputs[_transformed_name(_LABEL_KEY)] = tf.cast(inputs[_LABEL_KEY], tf.int64)

    return outputs
