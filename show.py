import re
import sys

file = sys.argv[1]
with open(file) as f:
    logs = [line.strip() for line in f]

current_index = 0
while current_index < len(logs):
    source_domain = None
    target_domain = None
    dstore_config = None
    valid_bleu = []
    test_bleu = []
    epochs = []

    while current_index < len(logs):
        if 'source_domain:' in logs[current_index]:
            if source_domain != None:
                break
            else:
                source_domain = logs[current_index].split(': ')[1]

        if 'target domain:' in logs[current_index]:
            target_domain = logs[current_index].split(': ')[1]
        
        if 'dstore_config:' in logs[current_index]:
            dstore_config = logs[current_index].split(': ')[1]
        
        if 'for epoch =' in logs[current_index]:
            epoch = re.findall(r'epoch = (\d+)', logs[current_index], re.S)[0]
            epochs.append(int(epoch))
        
        if 'Generate valid with beam=4: ' in logs[current_index]:
            bleu = re.findall(r'BLEU = (\d+.\d+) ', logs[current_index], re.S)[0]
            valid_bleu.append(float(bleu))
            # print('valid', logs[current_index])
        
        if 'Generate test with beam=4: ' in logs[current_index]:
            bleu = re.findall(r'BLEU = (\d+.\d+) ', logs[current_index], re.S)[0]
            test_bleu.append(float(bleu))
            # print('test', logs[current_index])
        
        current_index += 1
    
    if epochs != []:
        print('source_domain:', source_domain)
        print('target_domain:', target_domain)
        print('dstore_config:', dstore_config)

        if valid_bleu == []:
            for e, t in zip(epochs, test_bleu):
                print(f'epoch = {e}, valid_bleu: xxxx - test_bleu: {t}')
        else:
            max_valid_bleu = max(valid_bleu)
            for e, v, t in zip(epochs, valid_bleu, test_bleu):
                if v == max_valid_bleu:
                    print(f'epoch = {e}, valid_bleu: {v} - test_bleu: {t}   *best*')
                else:
                    print(f'epoch = {e}, valid_bleu: {v} - test_bleu: {t}')