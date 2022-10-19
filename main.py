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
    docsets = []  # 所有文档的名字
    all_doc_words_list = []  # 所有文档的词汇列表
    dict = set()  # 全局词典
    docs_to_shinglesets = {}  # 文档转换成bucketID的集合
    docs_to_shinglesetsmatrix = {}  # 文档转换成特征矩阵
    hash_signatures = []  # 哈希签名矩阵

    # Step0: 数据集处理
    for i in range(1, num_of_doc + 1):  # 对所有文档进行处理
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
        for index in range(0, len(words) - k_shingle + 1):  # 采用k-shingling组成分词集合
            shingle = words[index]  # + " " + words[index + 1]
            temp_words.append(shingle)
        all_doc_words_list.append(temp_words)
        dict = dict.union(set(temp_words))

    # Step1: 文档 ——> shingling sets
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

    # Step2: 生成Min-Hash签名矩阵
    numElems = int(num_of_doc * (num_of_doc - 1) / 2)
    paraM = randomPickHashParameter(num_of_hash_functions)  # 随机生成Hash函数
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

    # Step3: 比较相似度：
    estMatrix = np.zeros([num_of_doc, num_of_doc])
    for i in range(0, num_of_doc):
        sig1 = hash_signatures[i]
        for j in range(i + 1, num_of_doc):
            sig2 = hash_signatures[j]
            count = 0
            for k in range(0, num_of_hash_functions):
                count = count + (sig1[k] == sig2[k])
            estMatrix[i][j] = count / num_of_hash_functions

    # Step4: 输出前五个最大相似度的文档
    print("基于", num_of_hash_functions, "个哈希函数计算所得最相似的文档排序如下:")
    for i in range(5):
        index = np.unravel_index(estMatrix.argmax(), estMatrix.shape)
        print("文档", list(index)[0], "与文档", list(index)[1], "相似度为", estMatrix.max())
        estMatrix[list(index)[0]][list(index)[1]] = 0
