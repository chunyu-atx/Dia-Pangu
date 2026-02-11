NUM_EPOCHS = 10        #Number of epochs to train for
BATCH_SIZE = 6       #Change this depending on GPU memory
NUM_WORKERS = 4       #A value of 0 means the main process loads the data
LEARNING_RATE = 2e-5
LOG_EVERY = 200       #iterations after which to log status during training
VALID_NITER = 200    #iterations after which to evaluate model and possibly save (if dev performance is a new max)
PRETRAIN_PATH = "/media/t1/zcy/dia-pangu/evaluate/bert-base-chinese-safetensors"  #path to pretrained model, such as BlueBERT or BioBERT
PAD_IDX = 0           #padding index as required by the tokenizer 

#CONDITIONS is a list of all 14 medical observations 
CONDITIONS = ['enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity',
              'lung_lesion', 'edema', 'consolidation', 'pneumonia', 'atelectasis',
              'pneumothorax', 'pleural_effusion', 'pleural_other', 'fracture',
              'support_devices', 'no_finding']
CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}
