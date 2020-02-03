import numpy as np
import torch
import os

# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=10, verbose=False):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement. 
#                             Default: False
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.tolerance = 0.02 # 2%

#     def __call__(self, val_loss, model, optimizer, epoch, save_path):

#         score = val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, optimizer, epoch, save_path)
#         elif score > self.best_score*(1+self.tolerance):
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, optimizer, epoch, save_path)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model, optimizer, epoch,save_path):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': val_loss
#             }, open(os.path.join(save_path, 'state.t7'), 'wb'))

#         #torch.save(model.state_dict(), 'checkpoint.pt')
#         self.val_loss_min = val_loss

class EarlyStopping(object):
    def __init__(self,delta=5,patience=5):
        self.delta = delta
        self.min_loss = 1000
        self.prev_min_loss = 1000
        self.early_stop = False
        self.patience = patience
        self.wait=0
        self.val_loss_min = np.Inf

    def check_early_stopping(self,train_loss,val_loss,model, optimizer, epoch,save_path):
        loss = abs(train_loss-val_loss)
        if( loss > self.delta*self.min_loss):
            if(self.wait>self.patience):
                self.early_stop = True
                # added by me since else it is always =5 once it stops early
                self.wait = 0
            else:
                print("[WARN] loss increased patience {} with loss {} and threshold {}".format(
                    self.wait,loss,self.min_loss))
                self.wait+=1
        else:
            if loss < self.min_loss:
                temp = self.min_loss
                self.min_loss = (self.prev_min_loss+loss)/2. 
                self.prev_min_loss=temp
            self.save_checkpoint(val_loss, model, optimizer, epoch, save_path)
            self.early_stop = False

    def save_checkpoint(self, val_loss, model, optimizer, epoch,save_path):
        '''Saves model when validation loss decrease.'''
        
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
            }, open(os.path.join(save_path, 'state.t7'), 'wb'))

        #torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss