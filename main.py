import random

import numpy as np

file_path = r"D:\\article\\"
num_of_doc = 20
k_shingle = 1
num_of_hash_functions = 20
maxBucketID = 4000


# h(x) = (m * x + n) % k
def randomPickHashParameter(hash_number):
    rand_list = []
    cnt = 0
    while cnt < hash_number:
        rand_index = random.randint(0, maxBucketID)
        while rand_index in rand_list:
            rand_index = random.randint(0, maxBucketID)
        rand_list.append(rand_index)
        cnt += 1
    return rand_list


if __name__ == '__main__':
    docsets = []  # all doc names
    all_doc_words_list = []  # all doc word lists
    dict = set()  # 全局词典
    docs_to_shinglesets = {}  # the set of doc -> bucketID
    docs_to_shinglesetsmatrix = {}  # the set of doc -> feature_matrix
    hash_signatures = []  # hash signature matrix

    # Step0: dataset preprocessing
    for i in range(1, num_of_doc + 1):  # for all docs
        words = []
        filename = "output-" + str(i) + ".txt"
        docsets.append(filename)
        f = open(file_path + filename, 'r', encoding='UTF-8')
        for line in f.readlines():
            line = line.strip('\n')
            for word in line.split(" "):
                words.append(word)
        words = [j for j in words if j != '']
        temp_words = []
        for index in range(0, len(words) - k_shingle + 1):  # k-shingling
            shingle = words[index]  # + " " + words[index + 1]
            temp_words.append(shingle)
        all_doc_words_list.append(temp_words)
        dict = dict.union(set(temp_words))

    # Step1: docs ——> shingling sets
    for i in range(1, num_of_doc + 1):
        words = all_doc_words_list[i - 1]
        doc_to_single_sihingle = set()
        for index in range(0, len(words)):
            bucket_number = list(dict).index(words[index])
            doc_to_single_sihingle.add(bucket_number)
        feature_matrix = np.empty(len(list(dict)))
        for value in doc_to_single_sihingle:
            feature_matrix[value] = 1
        docs_to_shinglesets["output-" + str(i) + ".txt"] = sorted(doc_to_single_sihingle)
        docs_to_shinglesetsmatrix["output-" + str(i) + ".txt"] = feature_matrix

    # Step2: generate Min-Hash signature matrix
    numElems = int(num_of_doc * (num_of_doc - 1) / 2)
    paraM = randomPickHashParameter(num_of_hash_functions)  # genearte Hash functions randomly
    paraN = randomPickHashParameter(num_of_hash_functions)
    for doc_name in docsets:
        shingleIDSet = docs_to_shinglesets[doc_name]
        feature_matrix = docs_to_shinglesetsmatrix[doc_name]
        hash_signature = []
        for i in range(0, num_of_hash_functions):
            minHashCode = len(list(dict)) + 1
            for j in range(feature_matrix.shape[0]):
                if feature_matrix[j] == 1:
                    hashCode = (paraM[i] * j + paraN[i]) % len(list(dict))
            for shingleID in shingleIDSet:
                hashCode = (paraM[i] * shingleID + paraN[i]) % len(list(dict))
                if feature_matrix[shingleID] == 1:
                    minHashCode = min(minHashCode, hashCode)
            hash_signature.append(minHashCode)
        hash_signatures.append(hash_signature)

    # Step3: compare similarities：
    estMatrix = np.zeros([num_of_doc, num_of_doc])
    for i in range(0, num_of_doc):
        sig1 = hash_signatures[i]
        for j in range(i + 1, num_of_doc):
            sig2 = hash_signatures[j]
            count = 0
            for k in range(0, num_of_hash_functions):
                count = count + (sig1[k] == sig2[k])
            estMatrix[i][j] = count / num_of_hash_functions

    # Step4: output the most similar doc pair
    print("Based on", num_of_hash_functions, "hash functions sorted by similarities")
    for i in range(5):
        index = np.unravel_index(estMatrix.argmax(), estMatrix.shape)
        print("Doc", list(index)[0], "and Doc", list(index)[1], "Similarity is", estMatrix.max())
        estMatrix[list(index)[0]][list(index)[1]] = 0
