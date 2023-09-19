cuda=1

nohup python -u eval.py --cuda ${cuda} > ./log/eval.log &
