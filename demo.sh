# Because the epoch setting is based on NGRAM=2, if NGRAM is changed, the number of epoch should be changed.

NGRAM=2
python process_data.py GoogleNews-vectors-negative300.bin
python initialize_filter.py --ngram $NGRAM mr.p
python cnn.py mr.p weights_${NGRAM}.pkl --ngram $NGRAM
