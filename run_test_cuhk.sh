#!/bin/bash

CONFIGPATH='./configs/pstr/pstr_r50_24e_cuhk.py'
MODELPATH='pstr_r50_cuhk-2fd8c1d2.pth'
OUTPATH='work_dirs/pstr_results.pkl'


CUDA_VISIBLE_DEVICES=0 python ./tools/test.py ${CONFIGPATH} ${MODELPATH}  --eval bbox --out ${OUTPATH}
echo '------------------------'
CUDA_VISIBLE_DEVICES=0 python ./tools/test_results_cuhk.py ${OUTPATH}
#echo '------------------------'
CUDA_VISIBLE_DEVICES=3 python ./tools/test_results_cuhk_cbgm.py ${OUTPATH}
