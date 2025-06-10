GLOBALLEN="10000" # the maximum sequence length of the model
MAXCTXLEN="5" # the maximum input context length
GENLEN="32" # the maximun generation length

SEED=42
DEVICE="0,1" # the GPU device id
TOPP="0.0" # top-p sampling, set to 0.0 for greedy decoding
GPUS=2 # number of gpus
FLAG="no" # set to "yes" to enable int4 quantization to load the model
# ALPHA=0

TESTFILE="fin|$1"
# bash run_group_decode_fileio.sh $SEED $DEVICE $TESTFILE $GLOBALLEN $MAXCTXLEN $GENLEN $TOPP $GPUS $FLAG $ALPHA

# For alpha values from 0.1 to 1.0 with step 0.1
for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
    ALPHA=$i
    echo "Running with ALPHA=$ALPHA"
    bash run_group_decode_fileio.sh $SEED $DEVICE $TESTFILE $GLOBALLEN $MAXCTXLEN $GENLEN $TOPP $GPUS $FLAG $ALPHA
done