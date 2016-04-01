#!/bin/bash

MF=""
NUSERS=""
NITEMS=""
FACDIMS=(1 5 10 15 25)
REGS=("0.001" "0.01" "0.1" "1" "10")
LEARNRATE="0.001"
SEED="1"

TRAIN=""
TEST=""
VAL=""

for dim in "${FACDIMS[@]}";do
  for reg in "${REGS[@]}";do
    echo $MF $NUSERS $NITEMS $dim 1000 $dim $SEED $reg $reg $LEARNRATE 0.0 0.0 \
      $TRAIN $TEST $VAL "null1 null1 null2 null1 null2 > mf_"$reg"_"$dim".txt"
  done
done


