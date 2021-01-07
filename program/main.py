from functools import reduce
from time import gmtime, strftime
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim
import pandas as pd
from torch.cuda import device
from torch.utils.data import TensorDataset, DataLoader

from dataloader import read_npy as read_daily_data
from EEGNet import WTFNet_CNN as EEGNet
from EEGNet import SE

DATASET='raw'
DATASET='aug'

def get_bci_dataloaders():
    #train_x, train_y, test_x, test_y = read_daily_data(dataset=DATASET)#read_bci_data()
    train_x, train_y, test_x, test_y = read_daily_data(dataset=DATASET)#read_bci_data()
    datasets = []
    for train, test in [(train_x, train_y), (test_x, test_y)]:
        train = torch.stack(
            [torch.Tensor(train[i]) for i in range(train.shape[0])]
        )
        test = torch.stack(
            [torch.Tensor(test[i:i+1]) for i in range(test.shape[0])]
        )
        datasets += [TensorDataset(train, test)]

    return datasets

def get_data_loaders(train_dataset, test_dataset):
    #train_dataset, test_dataset = get_bci_dataloaders()
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=16384, shuffle=False)
    return train_loader, test_loader


def showResult(title='', epochs=1000, **kwargs):
    plt.figure(figsize=(15, 7))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    for label, data in kwargs.items():
        plt.plot(range(len(data)), data, '--' if 'test' in label else '-', label=label)
    plt.ylim(0, 100)
    plt.xlim(0, epochs)
    points = [(-5, 87), (310, 87)]
    (xpoints, ypoints) = zip(*points)

    plt.plot(xpoints, ypoints, linestyle='--', color='black')

    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.show()


    
def main2():
    multi_task_flag = True
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x, train_y, test_x, test_y = read_daily_data(dataset=DATASET)
    channel = train_x.shape[2]

    nets1 = {
        #"SE_hswish": EEGNet(nn.Hardswish, sq=True).to(device),
        #"SE_elu": EEGNet(nn.ELU, sq=True).to(device),
        #"SE_gelu": EEGNet(nn.GELU, channel=channel, sq=True).to(device),
        "SE_elu": EEGNet(nn.ELU, channel=channel, sq=False).to(device),
        #"gelu": EEGNet(nn.GELU, channel=channel, sq=False).to(device),
        #"SE_relu": EEGNet(nn.ReLU, sq=True).to(device),
        #"SE_relu6": EEGNet(nn.ReLU6, sq=True).to(device),
        #"SE_leaky_relu": EEGNet(nn.LeakyReLU, sq=True).to(device),
        #"hswish": EEGNet(nn.Hardswish).to(device),
        #"elu": EEGNet(nn.ELU).to(device),
        #"relu": EEGNet(nn.ReLU).to(device),
        #"relu6": EEGNet(nn.ReLU6).to(device),
        #"EEG_leaky_relu": EEGNet(nn.LeakyReLU).to(device)
    }

    
    nets = nets1
    #nets = nets2
    
    #pkl_path = 'params205034_8_500_16384_lstm.pkl'
    #pkl_path = 'best_90.196_new_shape_raw_param.pkl'
    #nets['SE_elu'].load_state_dict(torch.load(pkl_path))
    # Training setting
    
    loss_fn = nn.CrossEntropyLoss()

    learning_rates = {0.002}#, 0.0025, 0.0028}

    optimizer = torch.optim.Adam
    optimizers = {
        key: optimizer(value.parameters(), lr=learning_rate, weight_decay=0.0001)
        for key, value in nets.items()
        for learning_rate in learning_rates
    }

    epoch_size = 1#1000
    batch_size = 128#32#1140#16384#32#16384

    acc, cur_time, loss_log = train(nets, epoch_size, batch_size, loss_fn, optimizers)
    df = pd.DataFrame.from_dict(acc)
    df.to_csv(DATASET+'_1000_cnn_woflood_acc'+str(cur_time)+'.csv')
    df.describe().to_csv(DATASET+'1000_cnn_hs_woflood_desc_'+str(cur_time)+'.csv')
    
    #df_loss = pd.DataFrame.from_dict(loss_log)
    #df_loss.to_csv('hs_woflood_loss'+str(cur_time)+'.csv')

    print(df)
    return df



y_hat = None
y_hh = None 
# This train is for demo and recording accuracy
def train(nets, epoch_size, batch_size, loss_fn, optimizers):
    global y_hat
    global y_hh
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainDataset, testDataset = get_bci_dataloaders()
    trainLoader, testLoader = get_data_loaders(trainDataset, testDataset)

    accuracy = {
        **{key + "_train": [] for key in nets},
        **{key + "_test": [] for key in nets}
    }
    loss_dict = {
        **{key + "_train": [] for key in nets},
        **{key + "_test": [] for key in nets}
    }
    #flood_dict = {
    #    **{key + "_train": [] for key in nets}
    #}
    for epoch in range(epoch_size + 1):
        
        train_loss = {key: 0.0 for key in nets}
        test_loss = {key: 0.0 for key in nets}
        #train_flood = {key: 0.0 for key in nets}
        #test_flood = {key: 0.0 for key in nets}
        train_correct = {key: 0.0 for key in nets}
        test_correct = {key: 0.0 for key in nets}
        for step, (x, y) in enumerate(trainLoader):
            x = x.to(device)
            y = y.to(device).long().view(-1)

            for key, net in nets.items():
                b=0.01
                net.train(mode=True)
                y_hat = net(x)
                loss = loss_fn(y_hat, y)
                #flood = (loss-b).abs()+b
                #flood.backward()
                loss.backward()
                train_loss[key] += loss.item()
                #train_flood[key] += flood.item()
                train_correct[key] += (torch.max(y_hat, 1)[1] == y).sum().item()

            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad()
        
        
        with torch.no_grad():
            for step, (x, y) in enumerate(testLoader):
                x = x.to(device)
                y = y.to(device).long().view(-1)
                for key, net in nets.items():
                    net.eval()
                    y_hat = net(x)
                    loss = loss_fn(y_hat, y)
                    test_loss[key] += loss.item()
                    test_correct[key] += (torch.max(y_hat, 1)[1] == y).sum().item()
                    acc = ((torch.max(y_hat, 1)[1] == y).sum().item()* 100.0) / len(testDataset)
                    if acc > 88: 
                        cur_time = strftime("%H%M%S", gmtime())
                        #torch.save(net, 'Wafer-best-model.pkl')
                        torch.save(net.state_dict(), str(DATASET)+'-cnn-best-param'+str(cur_time)+'_'+str(acc)+'.pkl')
                        print(str(DATASET)+'-1000-cnn-best-param'+str(cur_time)+'_'+str(acc)+'.pkl')

                    
        for key, value in train_correct.items():
            accuracy[key + "_train"] += [(value * 100.0) / len(trainDataset)]
            
        for key, value in test_correct.items():
            accuracy[key + "_test"] += [(value * 100.0) / len(testDataset)]
            
        for key, value in train_loss.items():
            loss_dict[key + "_train"] += [value / len(trainLoader)]
        
        for key, value in test_loss.items():
            loss_dict[key + "_test"] += [value / len(testLoader)]
            
        #for key, value in train_flood.items():
        #    flood_dict[key + "_train"] += [value / len(trainLoader)]
        
        if epoch % 100 == 0:
            print('epoch : ', epoch, ' loss : ', loss.item())
            print(pd.DataFrame.from_dict(accuracy).iloc[[epoch]])
            #display(pd.DataFrame.from_dict(accuracy).iloc[[epoch]])
            print('')
            #print(y_hat)
            #pd.DataFrame(y_hat).to_csv('stock16_y_hat_'+str(epoch)+'.csv')
            #y_hh = pd.DataFrame(torch.max(y_hat, 1)[1])
            #display(y_hh)
            #y_hh.hist()
            #input()
        torch.cuda.empty_cache()
    showResult(title='Activation function comparison(EEGNet)'.format(epoch + 1), epochs=epoch_size, **accuracy)
    cur_time = strftime("%H%M%S", gmtime())
    torch.save(net.state_dict(), '1000-params'+str(cur_time)+'_8_1000_16384_2lstm_'+DATASET+'_cnn'+str(epoch)+'.pkl')
    print("params:", sum(p.numel() for p in net.parameters() if p.requires_grad))
    return accuracy, cur_time, loss_dict

df = main2()
print(df.describe())
