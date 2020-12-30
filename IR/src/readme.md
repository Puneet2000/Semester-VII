## IR Based Chatbot: by tuning BERT architecture
Team:
	- Puneet Mangla (CS17BTECH11029)
	- Yash Khasbarge (CS17BTECH11044)

### Libraries
- PyTorch
- Python3
- Hugging face Transformers

### Dataset
Genrate train, val, test splits without stemming and save them in ```./datasets``` folder

### Usage

1. **Modifying Vocabulary**
	- Create an ```input.txt``` that contains all text from train.csv file
	- Run ```python3 vocab.py``` to generate UDC vocabulary as ```udc_vocab.txt```
	- Disjoint tokens are in ```remain.txt```. You will have to concatenate it with actual BERT vocab. 
	- The final 50K vocab is already provided in ```bert_vocab.txt```
2. **Training**
	- With BERT: ```python3 main.py --dataset_root ./datasets --split train --batch-size 64 --cuda 0 --dataset_root splits --exp_name bert_full_finetune```
	- With ALBERT: ```python3 train_albert.py --dataset_root ./datasets --split train --batch-size 4 --cuda 0 --do_train --exp_name albert_finetune```
3. **Evaluation**
	- With BERT: ```python3 eval.py --dataset_root ./datasets --split test --exp_name bert_full_finetune```
	- With ALVERT: ```python3 train_albert.py --dataset_root ./datasets --split test --do_eval --resume --iter 20 --cuda 0 --exp_name albert_finetune```