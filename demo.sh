NGRAM=1
pythozn process_data.py ../GoogleNews-vectors-negative300.bin
python initialize_filter.py --ngram $NGRAM mr.p
python cnn.py mr.p weights_${NGRAM}.pkl