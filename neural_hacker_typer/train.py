from __future__ import print_function
import argparse
import os
import sys
import time
import random
import string

from typing import getch

import torch
import torch.nn as nn
from torch.autograd import Variable

from char_rnn import CharRNN

class ProgressBar(object):
     def __init__(self, total=100, stream=sys.stderr):
         self.total = total
         self.stream = stream
         self.last_len = 0
         self.curr = 0

     def count(self):
         self.curr += 1
         self.print_progress(self.curr)

     def print_progress(self, value):
         self.stream.write('\b' * self.last_len)
         self.curr = value
         pct = 100 * self.curr / self.total
         out = '{:.2f}% [{}/{}] \r'.format(pct, self.curr, self.total)
         self.last_len = len(out)
         self.stream.write(out)
         self.stream.flush()

def random_training_set(chunk_len, batch_size, file, args):
     '''
     TODO: Convert to stateful LSTM with more features
     '''
     inp = torch.LongTensor(batch_size, chunk_len)
     target = torch.LongTensor(batch_size, chunk_len)
     file_len = len(file)

     for bi in range(batch_size):

          start_index = random.randint(0, file_len - chunk_len)
          end_index = start_index + chunk_len + 1
          chunk = file[start_index:end_index]
          if args.debug:
               print ('chunk', chunk)
          inp[bi] = char_tensor(chunk[:-1])
          target[bi] = char_tensor(chunk[1:])

     inp = Variable(inp)
     target = Variable(target)

     if args.cuda:
          inp = inp.cuda()
          target = target.cuda()

     if args.debug:
          print (inp, target)
     return inp, target


def train_on_batch(inp, target, args):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda: hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()
    return loss.data[0] / args.chunk_len

def save(args):
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)



class Generator(object):
     '''
     Class to encapsulate generator functionality
     '''
     def __init__(self, decoder):
          self.decoder = decoder

     def generate(self, *args, **kwargs):
          raise NotImplementedError


class SimpleGenerator(Generator):

     def generate(self,
                  prime_str='int ',
                  predict_len=100,
                  temperature=0.1,
                  cuda=False,
                  args=None,
                  hidden=None):

          prime_input = Variable(char_tensor(prime_str).unsqueeze(0))
          
          if not hidden:
               hidden = decoder.init_hidden(1)
               prime_input = Variable(char_tensor(prime_str).unsqueeze(0))
          
               if cuda:
                    hidden = hidden.cuda()
                    prime_input = prime_input.cuda()        
                    # Use priming string to "build up" hidden state
               for p in range(len(prime_str) - 1):
                    _, hidden = decoder(prime_input[:,p], hidden)        
               
          predicted = ''
          inp = prime_input[:,-1]
          p_list = []


          for p in range(predict_len):
               output, hidden = decoder(inp, hidden)        
               # Sample from the network as a multinomial distribution
               output_dist = output.data.view(-1).div(temperature).exp()
               top_i = torch.multinomial(output_dist, 1)[0]
               p_list.append(top_i)
               # Add predicted character to string and use as next input
               predicted_char = all_characters[top_i]
        
               predicted += predicted_char
               inp = Variable(char_tensor(predicted_char).unsqueeze(0))
               if cuda: inp = inp.cuda()

          # print (p_list)
          return predicted, hidden
          
    
def generate(decoder,
             prime_str='int ',
             predict_len=100,
             temperature=0.35,
             cuda=False,
             args=None,
             hidden=None):

     prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

     if not hidden:
          hidden = decoder.init_hidden(1)
          prime_input = Variable(char_tensor(prime_str).unsqueeze(0))
          
          if cuda:
               hidden = hidden.cuda()
               prime_input = prime_input.cuda()        
          # Use priming string to "build up" hidden state
          for p in range(len(prime_str) - 1):
               _, hidden = decoder(prime_input[:,p], hidden)        

     predicted = ''
     inp = prime_input[:,-1]
     p_list = []


     for p in range(predict_len):
          output, hidden = decoder(inp, hidden)        
          # Sample from the network as a multinomial distribution
          output_dist = output.data.view(-1).div(temperature).exp()
          top_i = torch.multinomial(output_dist, 1)[0]
          p_list.append(top_i)
          # Add predicted character to string and use as next input
          predicted_char = all_characters[top_i]
        
          predicted += predicted_char
          inp = Variable(char_tensor(predicted_char).unsqueeze(0))
          if cuda: inp = inp.cuda()

     # print (p_list)
     return predicted, hidden


def generate_token(decoder,
                   prime_str='int ',
                   temperature=0.35,
                   cuda=False,
                   args=None,
                   init_hidden=None):

     prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

     if not init_hidden:
          hidden = decoder.init_hidden(1)
          prime_input = Variable(char_tensor(prime_str).unsqueeze(0))
          
          if cuda:
               hidden = hidden.cuda()
               prime_input = prime_input.cuda()        
          # Use priming string to "build up" hidden state
          for p in range(len(prime_str) - 1):
               _, hidden = decoder(prime_input[:,p], hidden)        
          init_hidden = hidden
     init_inp = prime_input[:,-1]

     is_good = False
     while (not is_good):
          is_good = True
          predicted = ''
          p_list = []
          hidden = init_hidden
          inp = init_inp
          stopped = False

          while (not stopped):
               print ('generate_token', inp [:10], hidden [:10])
               output, hidden = decoder(inp, hidden)        
               print ('output', output[:10])
               raise Exception
               # Sample from the network as a multinomial distribution
               output_dist = output.data.view(-1).div(temperature).exp()
               top_i = torch.multinomial(output_dist, 1)[0]
               try:
                    if top_i == p_list[-1] and top_i == p_list[-2]:
                         is_good = False
               except:
                    pass
               p_list.append(top_i)
               
               # Add predicted character to string and use as next input
               predicted_char = all_characters[top_i]          
               if predicted_char in string.whitespace:
                    stopped = True
               predicted += predicted_char
               print ('predicted', predicted)
               inp = Variable(char_tensor(predicted_char).unsqueeze(0))
               if cuda: inp = inp.cuda()

          if len(predicted) > 15:
               is_good = False
          
          
     # print (p_list)
     return predicted, hidden

    
# Initialize models and start training

def build_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--filename', type=str)
    argparser.add_argument('--n_epochs', type=int, default=2000)
    argparser.add_argument('--print_every', type=int, default=1)
    argparser.add_argument('--hidden_size', type=int, default=256)
    argparser.add_argument('--n_layers', type=int, default=3)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    argparser.add_argument('--chunk_len', type=int, default=100)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--debug', default=False)
    argparser.add_argument('--type', default=False, action='store_true')
    args = argparser.parse_args()

    if args.cuda:
        print("Using CUDA")
    return args


def read_file(filename):
    file = open(file)
    return file, len(file)


def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

if __name__ == '__main__':
    args = build_parser()

    
    
    SYMBOL_TABLE = os.path.join('../saved_model', 'vocab.sym')
    if args.type and os.path.exists(SYMBOL_TABLE):
         all_characters = list(set(open(SYMBOL_TABLE).read()))
    else:
         file = open(args.filename).read()
         print('Loaded file', args.filename)
         print('File length', len(file)/80, 'lines')
         all_characters = list(set(file))    
         with open(SYMBOL_TABLE, 'w') as vocab:
              print("".join(all_characters), file=vocab)
         
    n_characters = len(all_characters)
        
    decoder = CharRNN(n_characters, args.hidden_size,
                      n_characters, n_layers=args.n_layers)

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if args.type:
         # Enter typing mode
         print ('Typing Mode...')

         decoder = torch.load('../saved_model/linux.pt')         
         from typing import build_getch


         with build_getch() as getch:
              try:
                   getchar = getch()
                   hidden = None
                   generator = SimpleGenerator(decoder)
                   prime_text = 'struct'
                   sys.stdout.write(prime_text)

                   while(getchar!='~'):
                        #output_text, hidden = generate(decoder, prime_text, 20,
                        #                               cuda=args.cuda, args=args,
                        #                               hidden=hidden)
                        output_text, hidden = generator.generate(prime_text, 20,
                                                                 cuda=args.cuda, args=args,
                                                                 hidden=hidden)
                        sys.stdout.write(output_text)
                        prime_text += output_text
                        getchar = getch()
                        if len(prime_text) > 100:
                             prime_text = prime_text[-100:]
                   getch.reset()

              except (KeyboardInterrupt, Exception) as e:
                   getch.reset()
                   print (e.message)
                   raise e
              
         raise Exception('Exit!')
    
    else: # Train model
         if args.cuda: decoder.cuda()
         start = time.time()
         all_losses = []
         loss_avg = 0

         try:
              SAMPLES_PER_EPOCH = 10000
              total_samples = 0
              print("Training for %d epochs..." % args.n_epochs)
              for epoch in range(1, args.n_epochs + 1):
                   samples_processed = 0
                   progress_bar = ProgressBar(SAMPLES_PER_EPOCH)
                   while(samples_processed) < SAMPLES_PER_EPOCH:            
                        inp, target = random_training_set(args.chunk_len,
                                                          args.batch_size,
                                                          file, args)                        
                   loss = train_on_batch(inp, target, args)
                   samples_processed += args.batch_size
                   progress_bar.print_progress(samples_processed)
                   total_samples += samples_processed               
              if epoch % args.print_every == 0:
                   def time_since(start):
                        return time.time() - start
                   
                   print('[elapsed : %s epoch (%d %d%%) loss%.4f]' % \
                         (time_since(start), epoch,
                          epoch / args.n_epochs * 100, loss_avg/float(samples_processed)))

                   text, hidden = generate(decoder, 'int', 1000,
                                           cuda=args.cuda, args=args)
                   print(text)
                   print("Epoch {} : Saving...".format(epoch))
                   save(args)

         except KeyboardInterrupt:
              print("Saving before quit...")
              save(args)

