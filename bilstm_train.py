from preprocessing import preprocess_pipeline, reload_preprocessed

load_trained = False

if not load_trained:
    preprocess_pipeline("res/train_5500.label")

labels, sentences, vocabulary, vocabulary_embed, sentence_representation, label_index, label_representation = reload_preprocessed()

