from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import argparse
from nltk.translate.bleu_score import sentence_bleu
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
from evaluator.Bleu import cal_bleu

def SVC_(kernel="rbf", gamma=1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("linearSVC", svm.SVC(kernel="rbf", gamma=gamma))
    ])

if __name__ == "__main__":
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    # 通过valid来确定svm分割线
    parser.add_argument('-valid_retrieval_msg', required=True)
    parser.add_argument('-valid_retrieval_bleu', required=True)
    parser.add_argument('-valid_generate_msg', required=True)
    parser.add_argument('-valid_generate_score', required=True)
    parser.add_argument('-ground_truth', required=True)
    
    # 在test上使用
    parser.add_argument('-test_retrieval_msg', required=True)
    parser.add_argument('-test_retrieval_bleu', required=True)
    parser.add_argument('-test_generate_msg', required=True)
    parser.add_argument('-test_generate_score', required=True)
    parser.add_argument('-test_ground_truth', required=True)
    
    parser.add_argument('-output', required=True)

    opt = parser.parse_args()
    
    valid_retrieval_msg, valid_retrieval_bleu, valid_generate_msg, valid_generate_score, ground_truth = [], [], [], [], []
    with open(opt.valid_retrieval_msg, 'r') as f:
        for line in f.readlines():
            valid_retrieval_msg.append(line)
    with open(opt.valid_retrieval_bleu, 'r') as f:
        for line in f.readlines():
            valid_retrieval_bleu.append(line)
    with open(opt.valid_generate_msg, 'r') as f:
        for line in f.readlines():
            valid_generate_msg.append(line.split('\t')[1])
    with open(opt.valid_generate_score, 'r') as f:
        for line in f.readlines():
            valid_generate_score.append(line)
    with open(opt.ground_truth, 'r') as f:
        for line in f.readlines():
            ground_truth.append(line.split('\t')[1])
    assert len(valid_retrieval_msg) == len(valid_retrieval_bleu) == len(valid_generate_msg) == len(valid_generate_score) == len(ground_truth)

    # find best threshold
    best = [0]
    for aa in range(50, 90, 1):
        for bb in range(-30, -80, -1):
            a = aa / 100.0
            b = bb / 100.0
            with open(opt.output, 'w') as f:
                x, y = [], []
                for i, j, k, l, m in zip(valid_retrieval_msg, valid_retrieval_bleu, valid_generate_msg, valid_generate_score, ground_truth):
                    if float(j) < a or float(l) < b:
                        continue
                    ref = [m.split()]
                    b1 = sentence_bleu(ref, i.split())
                    b2 = sentence_bleu(ref, k.split())
                    if b1 < 0.001 and b2 < 0.001:
                        continue
                    elif b1 > b2:
                        x.append([float(j), float(l)])
                        y.append(1)
                    else:
                        x.append([float(j), float(l)])
                        y.append(-1)
                x.append([1, b])
                y.append(1)
                x.append([a, 0])
                y.append(-1)
                clf = SVC_(kernel="rbf", gamma=20)
                clf.fit(x, y)

                for i, j, k, l in zip(valid_retrieval_msg, valid_retrieval_bleu, valid_generate_msg, valid_generate_score):
                    if float(j) < a:
                        f.write(k)
                        continue
                    elif float(l) < b:
                        f.write(i)
                        continue
                    res = clf.predict(np.array([float(j), float(l)]).reshape(1, -1))
                    if res > 0:
                        f.write(i)
                    else:
                        f.write(k)
            bleu = cal_bleu(opt.output, opt.ground_truth)
            if bleu > best[0]:
                best = [bleu, a, b, clf]
                # print(best)
        #         break
        # if best[0] > 0: 
        #     break

    # print(best)
    test_retrieval_msg, test_retrieval_bleu, test_generate_msg, test_generate_score = [], [], [], []
    with open(opt.test_retrieval_msg, 'r') as f:
        for line in f.readlines():
            test_retrieval_msg.append(line)
    with open(opt.test_retrieval_bleu, 'r') as f:
        for line in f.readlines():
            test_retrieval_bleu.append(line)
    with open(opt.test_generate_msg, 'r') as f:
        for line in f.readlines():
            test_generate_msg.append(line.split('\t')[1])
    with open(opt.test_generate_score, 'r') as f:
        for line in f.readlines():
            test_generate_score.append(line)
    assert len(test_retrieval_msg) == len(test_retrieval_bleu) == len(test_generate_msg) == len(test_generate_score)
    with open(opt.output, 'w') as f:
        for i, j, k, l in zip(test_retrieval_msg, test_retrieval_bleu, test_generate_msg, test_generate_score):
            if float(j) < best[1]:
                f.write(k)
                continue
            elif float(l) < best[2]:
                f.write(i)
                continue
            res = best[3].predict(np.array([float(j), float(l)]).reshape(1, -1))
            if res > 0:
                f.write(i)
            else:
                f.write(k)
    print(best[1:3])
    bleu = cal_bleu(opt.output, opt.test_ground_truth)
    print(bleu)
                    


    
    
    
    
    # a = 0.99
    # # for bb in range(-40, -19, 1):
    # b = -0.37
    # # for cc in range(60, 100, 1):
    # # for dd in range(-80, -40, 1):
    # # b = bb / 100.0
    #     # c = cc / 100.0
    #     # d = dd / 100.0
    # c = 0.8
    # d = -0.6
    # # a, b, c, d = 0.99, -0.1, 0.8, -0.6
    # # a, b, c, d = 0.99, -3, 0.7, -5
    # print(a, b, c, d)
    # # 生成svm数据
    # x, y = [], []
    # # with open('result.csv', 'w') as f:
    
    #     # print(float(l) / float(j))
    #     if float(l) > c or float(j) > a:
    #         continue
    #     ref = [m.split()]
    #     b1 = sentence_bleu(ref, i.split())
    #     b2 = sentence_bleu(ref, k.split())
    #     # if float(j) > a:
    #     #     continue
    #     # elif float(l) > b:
    #     #     continue
    #     # elif float(j) < c:
    #     #     continue
    #     # elif float(l) < d:
    #     #     continue
    #     if b1 < 0.001 and b2 < 0.001:
    #         continue
    #         # y.append(0)
    #     elif b1 > b2:
    #         x.append([float(j), float(l)])
    #         y.append(1)
    #     else:
    #         x.append([float(j), float(l)])
    #         y.append(-1)
    # x.append([a, d])
    # y.append(1)
    # x.append([c, b])
    # y.append(-1)
    #         # f.write(str(b1) + ',' + str(b2) + ',' + (','.join([i.replace(',', '，'), j, k.replace(',', '，'), l, m.replace(',', '，')]).replace('\n', ' ')) + '\n')
        

    

    # clf = SVC_(kernel="rbf", gamma=20)
    # clf.fit(x, y)
    # # clf = svm.SVC(kernel='poly', degree=5).fit(x, y)
    # # clf = svm.SVC(kernel='linear').fit(x, y)

    # # clf = svm.SVC(kernel='rbf').fit(x, y)
    
    # test_retrieval_msg, test_retrieval_bleu, test_generate_msg, test_generate_score = [], [], [], []
    # with open(opt.test_retrieval_msg, 'r') as f:
    #     for line in f.readlines():
    #         test_retrieval_msg.append(line)
    # with open(opt.test_retrieval_bleu, 'r') as f:
    #     for line in f.readlines():
    #         test_retrieval_bleu.append(line)
    # with open(opt.test_generate_msg, 'r') as f:
    #     for line in f.readlines():
    #         test_generate_msg.append(line)
    # with open(opt.test_generate_score, 'r') as f:
    #     for line in f.readlines():
    #         test_generate_score.append(line)
    # assert len(test_retrieval_msg) == len(test_retrieval_bleu) == len(test_generate_msg) == len(test_generate_score)
    # count = 0
    # with open(opt.output, 'w') as f:
    #     for i, j, k, l in zip(test_retrieval_msg, test_retrieval_bleu, test_generate_msg, test_generate_score):
    #         # if float(l) > b:
    #         #     f.write(k)
    #         #     continue
    #         # elif float(j) > a:
    #         #     f.write(i)
    #         #     # print(1)
    #         #     continue
    #         if float(j) < c:
    #             f.write(k)
    #             continue
    #         elif float(l) < d:
    #             f.write(i)
    #             continue
    #         res = clf.predict(np.array([float(j), float(l)]).reshape(1, -1))
    #         count += 1
    #         # f.write(k)
    #         if res > 0:
    #             f.write(i)
    #         else:
    #             f.write(k)
    # os.system("python ./evaluation/Bleu-B-Norm.py /root/edist1/data/Codisum/cleaned.test.msg < /root/edist1/codet5_svm > bleu")
    

    # # print(count)
    # # 设置子图数量
    # fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(7,7))
    # ax0, ax1 = axes.flatten()
    # for i, j in zip(x, y):
    #     if j == 1:
    #         ax0.scatter(i[0],i[1],c='r',marker='*')
    #     # elif j == 0:
    #     #     ax0.scatter(i[0],i[1],c='b',marker='*')
    #     else:
    #         ax0.scatter(i[0],i[1],c='g',marker='*')
    
    # plt.gcf().set_size_inches(20, 12)
    # plt.savefig("origin.png", dpi=300) 

    
# python svm.py \
#     -valid_retrieval_msg ${DATAPATH}/cleaned.test.msg.nngen \
#     -valid_retrieval_bleu ${DATAPATH}/cleaned.test.msg.bleu4 \
#     -valid_generate_msg ${SAVEPATH}/result/${LAYER}layer_${HEAD}head_${MASK_RATE}mask.msg \
#     -valid_generate_score ${SAVEPATH}/result/${LAYER}layer_${HEAD}head_${MASK_RATE}mask.msg_score \
#     -ground_truth ${DATAPATH}/cleaned.test.msg \
#     -test_retrieval_msg ${DATAPATH}/cleaned.test.msg.nngen \
#     -test_retrieval_bleu ${DATAPATH}/cleaned.test.msg.bleu4 \
#     -test_generate_msg ${SAVEPATH}/result/${LAYER}layer_${HEAD}head_${MASK_RATE}mask.msg \
#     -test_generate_score ${SAVEPATH}/result/${LAYER}layer_${HEAD}head_${MASK_RATE}mask.msg_score \
#     -output  ${SAVEPATH}/result/${LAYER}layer_${HEAD}head_${MASK_RATE}mask.msg_svm \
#     >> ${SAVEPATH}/log/train_log