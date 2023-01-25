#!/bin/sh

# MODELS=$(ls output)
# for MODEL in $MODELS
# do
#     echo "" >> output/result
#     echo "#===================#" >> output/result
#     echo $MODEL >> output/result
    
#     python3 evaluate.py \
#         --refs_filename /root/edist1/data/Codisum/cleaned.test.msg \
#         --preds_filename ./output/${MODEL} \
#         >> output/result
#     # python ./evaluator/Bleu-B-Norm.py \
#     #     /root/edist1/data/Codisum/cleaned.test.msg \
#     #     < ./output/${MODEL} \
#     #     >> output/result

#     # python ./evaluator/evaluate.py \
#     #     /root/edist1/data/Codisum/cleaned.test.msg \
#     #     ./output/${MODEL} \
#     #     >> output/result
# done

# MODEL=race
# python ./evaluator/Bleu-B-Norm.py \
#         /root/edist1/data/Codisum/cleaned.test.msg \
#         < ./output/${MODEL} \
#         >> output/result

#     python ./evaluator/evaluate.py \
#         /root/edist1/data/Codisum/cleaned.test.msg \
#         ./output/${MODEL} \
#         >> output/result

# echo "cpp"
# python3 evaluate.py --refs_filename /root/CodeT51/exp/cpp/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/exp/cpp/prediction/s2/test_best-bleu.output1
# echo "csharp"
# python3 evaluate.py --refs_filename /root/CodeT51/exp/csharp/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/exp/csharp/prediction/s2/test_best-bleu.output1
# echo "java"
# python3 evaluate.py --refs_filename /root/CodeT51/exp/java1/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/exp/java1/prediction/s2/test_best-bleu.output1
# python3 evaluate.py --refs_filename /root/CodeT51/exp/java1/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/exp/java1/prediction/svm
# echo "javascript"
# python3 evaluate.py --refs_filename /root/CodeT51/exp/javascript/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/exp/javascript/prediction/s2/test_best-bleu.output1
# echo "python"
# python3 evaluate.py --refs_filename /root/CodeT51/exp/python/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/exp/python/prediction/s2/test_best-bleu.output1

# echo "cpp"
# python3 evaluate.py --refs_filename /root/CodeT51/exp/java/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/exp/java/prediction/svm
# echo "csharp"
# python3 evaluate.py --refs_filename /root/CodeT51/exp/csharp/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/output/MCMD/csharp/3layer_9head.msg_join
# echo "java"
# python3 evaluate.py --refs_filename /root/CodeT51/exp/java1/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/output/MCMD/java1/3layer_9head.msg_join
# echo "javascript"
# python3 evaluate.py --refs_filename /root/CodeT51/exp/javascript/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/output/MCMD/javascript/3layer_9head.msg_join
# echo "python"
# python3 evaluate.py --refs_filename /root/CodeT51/exp/python/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/output/MCMD/python/3layer_9head.msg_join

date=$1
# python -W ignore svm.py \
#     -valid_retrieval_msg /root/CodeT51/exp/$date/output/retrieval/test.output \
#     -valid_retrieval_bleu /root/CodeT51/exp/$date/output/retrieval/test.score \
#     -valid_generate_msg /root/CodeT51/exp/$date/prediction/s2/test_best-bleu.output \
#     -valid_generate_score /root/CodeT51/exp/$date/prediction/s2/test_best-bleu.score \
#     -ground_truth /root/CodeT51/exp/$date/prediction/s2/test_best-bleu.gold \
#     -test_retrieval_msg /root/CodeT51/exp/$date/output/retrieval/test.output \
#     -test_retrieval_bleu /root/CodeT51/exp/$date/output/retrieval/test.score \
#     -test_generate_msg /root/CodeT51/exp/$date/prediction/s2/test_best-bleu.output \
#     -test_generate_score /root/CodeT51/exp/$date/prediction/s2/test_best-bleu.score \
#     -test_ground_truth /root/CodeT51/exp/$date/prediction/s2/test_best-bleu.gold \
#     -output  /root/CodeT51/exp/$date/prediction/svm

python3 evaluate.py --refs_filename /root/CodeT51/exp/$date/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/exp/$date/prediction/svm
python3 evaluate.py --refs_filename /root/CodeT51/exp/$date/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/exp/$date/prediction/s2/test_best-bleu.output1
python3 evaluate.py --refs_filename /root/CodeT51/exp/$date/prediction/s2/test_best-bleu.gold1 --preds_filename /root/CodeT51/exp/$date/output/retrieval/test.output
