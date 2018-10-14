from myDataset import myDataset
from tripletLoss import TripletLoss
from utils import EER
from model import EmbeddingNet

import os
import argparse
import torch
import torchvision.models as models
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time



def train(train_loader, model, optimizer, criterion, epoch, args):
    start = time.time() 
    model.train()
    for step, (inputs, labels) in enumerate(train_loader):
        if args.cuda is True:
            inputs = inputs.cuda()
            labels = labels.cuda()
        print("read done") 
        optimizer.zero_grad()
        outputs = model.classifier_forward(inputs)
        print("output done")
        _, predicted = torch.max(outputs.data, 1)
        print("predicted label: {}, True label: {}".format(predicted, labels))
        loss = criterion(outputs, labels)
        print("loss at step{}: {}".format(step, loss.item()))
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            end = time.time()
            print("Train Epoch {}, Step {}, ({}%) in {}\nLoss: {}".format(
                epoch, step, 
                100 * step * len(inputs) / train_loader.dataset.__len__(),
                end-start, loss.item()))
        
        if step % args.checkpoint == 0:
            save_model(epoch, model, optimizer, loss, step, "./weights")

def save_model(epoch, model, optimizer, loss, step, save_path):
    filename = save_path + str(epoch) + '-' + str(step) + '-' + "%.6f" % loss.item() + '.pth'
    print('Save model at Train Epoch: {} [Step: {}\tLoss: {:.12f}]'.format(
        epoch, step, loss.item()))
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filename)

def load_model(epoch, step, loss, model, optimizer, save_path):
    filename = save_path + str(epoch) + '-' + str(step) + '-' + str(loss) + '.pth'
    if os.path.isfile(filename):
        print("######### loading weights ##########")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        print('########## loading weights done ##########')
        return model, optimizer, start_epoch, loss
    else:
        print("no such file: ", filename)    


def embed_utterances(dev_loader, model, flag):
    dev_loader.embedding_flag = flag
    embeddings_array = []
    with torch.no_grad():
        for step, inputs in enumerate(dev_loader):
            if args.cuda is True:
                inputs = inputs.cuda()
            embeddings = model(inputs)
            embeddings_array.append(embeddings[0])
            print("Embed {} utterances".format(step / dev_loader.__len__()))
            print(len(embeddings_array))
    return embeddings_array

def dev(dev_loader, model, epoch):
    start = time.time()
    model.eval()
    # Embed all the utterances in the enrollment array
    print("########## Start Embedding Enrol ##########")
    enrols_array = embed_utterances(dev_loader, model, "enrol")
    print("########## Embed Enrol Done ##########")
    print("########## Start Embedding Test ##########")
    # Embed all the utterances in the test array
    tests_array = embed_utterances(dev_loader, model, "test")
    print("########## Embed Test Done ##########")
    # start testing
    dev_loader.embedding_flag = "trail"
    with torch.no_grad():
        for step, (enrol_idx, test_idx, label) in enumerate(dev_loader):
            scores = []
            true_labels = []
            if args.cuda is True:
                enrol_idx = enrol_idx.cuda()
                test_idx = test_idx.cuda()
                label = label.cuda()
            
            enrol_embedding = model(enrols_array[enrol_idx])
            test_embedding = model(tests_array[test_idx])
            cos = nn.CosineSimilarity(dim=1)
            similarity_score = cos(enrol_embedding, test_embedding)
            scores.append(similarity_score)
            true_labels.append(label)
            print("True label {}, Similarity score {}".format(label, similarity_score))
            if step % args.log_interval == 0:
                end = time.time()
                print("Epoch {}, Step {}, ({}%) in {}".format(
                    epoch, step, 
                    100 * step / dev_loader.dataset.__len__(),
                    end-start))
        eer, threshold = EER(true_labels, scores)
        print("EER {}, threshole {}".format(eer, threshold))
    
def test(test_loader, model):
    pass


def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda is True:
        print("##########Running in CUDA GPU##########")
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        print("##########Running in CPU##########")
        kwargs = {}
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    train_data = myDataset("./data", [1], args.nframes, "train", None)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

    dev_data = myDataset("./data", None, args.nframes, "dev", "enrol")
    dev_loader = Data.DataLoader(dataset=dev_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # test_data = myDataset("./data", None, args.nframes, "test", "enrol")
    # test_loader = Data.DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    embeddingNet = EmbeddingNet(embedding_size=64, num_classes=train_data.nspeakers)
    embeddingNet = embeddingNet.double()
    if args.cuda:
        embeddingNet.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(embeddingNet.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, 5)
    start_epoch = 0

    for epoch in range(args.epochs):
        scheduler.step()
        train(train_loader, embeddingNet, optimizer, criterion, epoch, args)
        dev(dev_loader, embeddingNet, epoch)

    test(test_loader, embeddingNet)
    

def arguments():
    parser = argparse.ArgumentParser(description="Speaker Verificiation via CNN")
    parser.add_argument('--batch-size', type=int, default=4, metavar='BS',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='TBS',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='L2 regularization')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='L',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--checkpoint', type=int, default=50, metavar="R",
                        help='checkpoint to save model parameters')
    parser.add_argument('--resume', type=bool, default=False, metavar="R",
                        help='resume training from saved weight')
    parser.add_argument('--weights-path', type=str, default="./weights",
                        help='path to save weights')
    parser.add_argument('-load-epoch', type=str, default=0, metavar="LE",
                        help='number of epoch to be loaded')
    parser.add_argument('-load-step', type=str, default=0, metavar="LS",
                        help='number of step to be loaded')
    parser.add_argument('-load-loss', type=str, default=0, metavar="LL",
                        help='loss item to be loaded')
    parser.add_argument('-nframes', type=int, default=9000, metavar="N",
                        help='trim or pad utterances to number of frames')
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    main(args)
    
