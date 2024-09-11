#!/bin/bash

# eog ./your-image.png

for HH in 00 01 #02 03 04 05 06 07 08 09 10 11 12 #14 16 18 20 22
do
  for MM in $(seq -w 0 1 59) # 12 15 18 21 24 27 #$(seq -w 2 4 58) #01 10 15 20 25 30
  do
    for SS in 00 #$(seq -w 1 1 10) #10 20 30 40 50 #02 03 04 05 10 15 20
    do

      python3 ./scripts/plot2d_BSIM.py "T_temperature" ./BSIM_output/BSIM_checkpoint_2023-01-01_${HH}-${MM}-${SS}/
      mv T_temperature.png T_temperature_${HH}-${MM}-${SS}.png
      echo "Finish:: T_temperature_${HH}-${MM}-${SS}.png"


    done
  done
done

