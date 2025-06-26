#!/bin/bash

# Tạo thư mục cho VnCoreNLP
mkdir -p vncorenlp/models

# Tải VnCoreNLP
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar -O vncorenlp/VnCoreNLP-1.1.1.jar

# Tải models
cd vncorenlp/models
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
cd ../..