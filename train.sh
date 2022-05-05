# Change these as needed
DATASET=cnndm
MODEL=bart
METRIC=ctc_sum
NUM_BEAMS=8
DIVERSITY_PENALTY=0.8
NUM_RETURN_SEQUENCES=8

DATA_DIR=data/$DATASET/$MODEL
CKPT_DIR=checkpoints/$DATASET/$MODEL
if [ $MODEL == 'bart' ] && [ $DATASET == 'cnndm' ]; then
    HF_MODEL=facebook/bart-large-cnn
elif [ $MODEL == 'bart' ] && [ $DATASET == 'xsum' ]; then
    HF_MODEL=facebook/bart-large-xsum
elif [ $MODEL == 'pegasus' ] && [ $DATASET == 'cnndm' ]; then
    HF_MODEL=google/pegasus-cnn_dailymail
else
    HF_MODEL=google/pegasus-xsum
fi

# Sample training candidates
mkdir -p data/$DATASET/$MODEL
python run-summarizer.py --do_predict --predict_split=train --gpus=1 --dataset=DATASET --model_name_or_path=$MODEL --predictions_file=$DATA_DIR/diverse-samples-train.jsonl --num_beams=$NUM_BEAMS --num_beam_groups=$NUM_BEAMS --diversity_penalty=$DIVERSITY_PENALTY --num_return_sequences=$NUM_RETURN_SEQUENCES
# Sample validation candidates
python run-summarizer.py --do_predict --predict_split=validation --gpus=1 --dataset=DATASET --model_name_or_path=$MODEL --predictions_file=$DATA_DIR/diverse-samples-validation.jsonl --num_beams=$NUM_BEAMS --num_beam_groups=$NUM_BEAMS --diversity_penalty=$DIVERSITY_PENALTY --num_return_sequences=$NUM_RETURN_SEQUENCES
# Sample test candidates
python run-summarizer.py --do_predict --predict_split=test --gpus=1 --dataset=DATASET --model_name_or_path=$MODEL --predictions_file=$DATA_DIR/diverse-samples-test.jsonl --num_beams=$NUM_BEAMS --num_beam_groups=$NUM_BEAMS --diversity_penalty=$DIVERSITY_PENALTY --num_return_sequences=$NUM_RETURN_SEQUENCES

# Get the scores for the candidates
python scorer.py --source=$DATA_DIR/diverse-samples-train.jsonl --results_rouge=$DATA_DIR/results-rougel-train.jsonl --results_ctc=$DATA_DIR/results-ctc-train.jsonl --results_questeval=$DATA_DIR/results-questeval-train.jsonl
python scorer.py --source=$DATA_DIR/diverse-samples-validation.jsonl --results_rouge=$DATA_DIR/results-rougel-validation.jsonl --results_ctc=$DATA_DIR/results-ctc-validation.jsonl --results_questeval=$DATA_DIR/results-questeval-validation.jsonl
python scorer.py --source=$DATA_DIR/diverse-samples-test.jsonl --results_rouge=$DATA_DIR/results-rougel-test.jsonl --results_ctc=$DATA_DIR/results-ctc-test.jsonl --results_questeval=$DATA_DIR/results-questeval-test.jsonl

# Train the model
python run-ranker.py --do_train --gpus=1 --batch_size=24 --data_path=$DATA_DIR --metric=$METRIC --output=$CKPT_DIR