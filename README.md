# HEBBLSTM

## Environments
```
#requirements.txt

joblib==1.1.0
matplotlib==3.5.3
numpy==1.23.3
scipy==1.9.1
torch==1.12.1

```

## Usage
```bash
python script_simple.py 
--model_name HebbLSTM \ #['HebbNet', 'nnLSTM', 'HebbLSTM', 'DoubleHebb']
--input_dim 100 \ 
--hidden_dim 100 \ 
--train_mode multiR \ #['dat', 'inf', 'curr', 'multiR']
--R 3 \    # Used in 'dat' and 'inf' mode
--T 500 \ 
--threshold 4.9 # 0.98 * 5
```