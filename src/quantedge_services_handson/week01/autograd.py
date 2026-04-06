import torch

"""
Setup                                                                                                                                       

  prices = torch.tensor([10.0, 12.0, 11.0, 13.0, 15.0])                                                                                       
  #                       ↑     ↑     ↑     ↑     ↑                                                  
  # index:                0     1     2     3     4                                                                                           
                                                                                                                                              
  5 prices.                                                                                                                                   
                                                                                                                                              
  ---                                                                                                
  prices[1:] — "from index 1 to the end" 
                                                                                                                                              
  prices[1:]  →  tensor([12.0, 11.0, 13.0, 15.0])
  #                       ↑     ↑     ↑     ↑                                                                                                 
  # original index:       1     2     3     4                                                        
                                                                                                                                              
  Skips the first element. Length = 4.                                                                                                        
                                         
  ---                                                                                                                                         
  prices[:-1] — "from start to second-to-last"                                                       
                                         
  prices[:-1]  →  tensor([10.0, 12.0, 11.0, 13.0])
  #                        ↑     ↑     ↑     ↑                                                                                                
  # original index:        0     1     2     3
                                                                                                                                              
  Skips the last element. Length = 4.                                                                                                         
                                         
  ---                                                                                                                                         
  Why subtract them?                                                                                 
                                         
  Line them up:
                                                                                                                                              
  prices[1:]   =  [12.0, 11.0, 13.0, 15.0]    ← "next price"
  prices[:-1]  =  [10.0, 12.0, 11.0, 13.0]    ← "current price"                                                                               
                  ─────  ─────  ─────  ─────                                                                                                  
  prices[1:] - prices[:-1] = [+2.0, -1.0, +2.0, +2.0]                                                                                         
                                                                                                                                              
  Each element is next price minus current price = the price change between consecutive bars.                                                 
                                                                                                                                              
  ---                                                                                                                                         
  Compare to a loop (same thing)                                                                     
                                                                                                                                              
  deltas = []
  for i in range(len(prices) - 1):                                                                                                            
      deltas.append(prices[i+1] - prices[i])                                                         
  # deltas = [2.0, -1.0, 2.0, 2.0]       
                                                                                                                                              
  The slicing version prices[1:] - prices[:-1] does this in one vectorized operation — way faster than a loop, and it's the PyTorch idiom for 
  "consecutive differences."                                                                                                                  
                                                                                                                                              
  ---                                                                                                
  In our context                         
                
  For forex prices, deltas[i] = how much the price moved from minute i to minute i+1. Squaring and averaging gives you a measure of total
  volatility — that's the "loss" we backprop through.                                                                                         
   
"""

"""
 Line by line — with a 5-element example

  prices = torch.tensor([10.0, 12.0, 11.0, 13.0, 15.0])                                                                                       
                                                                                                                                              
  ---                                                                                                                                         
  Line 1: prices.requires_grad_(True)                                                                                                         
                                                                                                     
  What it does: Tells PyTorch "I'm going to do math with this tensor — please record every operation so I can compute gradients later."
                                                                                                                                              
  Before:                                                                                                                                     
  prices = tensor([10., 12., 11., 13., 15.])    requires_grad=False                                                                           
                                                                                                                                              
  After:                                                                                             
  prices = tensor([10., 12., 11., 13., 15.], requires_grad=True)                                                                              
                                                                
  The trailing underscore (requires_grad_ not requires_grad) means modify in-place — common PyTorch convention. From now on, every operation  
  involving prices is recorded in PyTorch's internal computation graph.                                                                       
                                         
  ---                                                                                                                                         
  Line 2: loss = (prices[1:] - prices[:-1]).pow(2).mean()                                            
                                                                                                                                              
  This is the forward pass. Four operations chained together. Let's expand each:
                                                                                                                                              
  Step 2a: prices[1:] - prices[:-1] → deltas                                                                                                  
  prices[1:]   = [12., 11., 13., 15.]      (skip first)                                                                                       
  prices[:-1]  = [10., 12., 11., 13.]      (skip last)                                                                                        
  ─────────────────────────────────────                                                                                                       
  deltas       = [ 2., -1.,  2.,  2.]      (4 elements)
                                                                                                                                              
  Step 2b: .pow(2) → square each delta                                                                                                        
  deltas²      = [ 4.,  1.,  4.,  4.]                                                                                                         
                                                                                                                                              
  Step 2c: .mean() → average → single number (scalar)                                                                                         
  loss = (4 + 1 + 4 + 4) / 4 = 13.0 / 4 = 3.25                                                       
                                                                                                                                              
  So loss = tensor(3.25, grad_fn=<MeanBackward0>).                                                                                            
                                                                                                                                              
  Why a scalar? Backward propagation can only start from a single number — you can't ask "what's the gradient of a vector?" without specifying
   which vector element.                                                                                                                      
                                                                                                                                              
  The graph PyTorch built behind the scenes:                                                                                                  
  prices ──► slice ──► sub ──► pow ──► mean ──► loss
                                                ↑                                                                                             
                                                (scalar, ready to backprop from)                                                              
                                                                                
  Every node in that graph remembers how to compute its derivative.                                                                           
                                                                                                                                              
  ---                                                                                                                                         
  Line 3: loss.backward()                                                                                                                     
                                                                                                                                              
  What it does: Walks the graph from loss backward to prices, applying the chain rule at each node, and writes the result into prices.grad.
                                                                                                                                              
  This is the moment autograd actually does work. It computes:                                       
  prices.grad[k] = ∂loss / ∂prices[k]    for every k                                                                                          
                                                                                                     
  For our 5 prices, it computes 5 gradient values — telling you "if I nudge prices[k] by a tiny amount, how much will loss change?"           
                                                                                                                                              
  Nothing is returned. The result is stored as a side effect in prices.grad.                                                                  
                                                                                                                                              
  ---                                                                                                                                         
  Line 4: return prices.grad                                                                         
                                         
  Before backward: prices.grad is None
  After backward: prices.grad is a tensor with the same shape as prices                                                                       
   
  For our example, prices.grad would be approximately:                                                                                        
  tensor([-1.0,  1.5, -1.5,  0.0,  1.0])                                                             
                                                                                                                                              
  Each element answers: "If I make this price slightly bigger, will the squared-difference loss go up (+) or down (−)?"                       
                                                                                                                                              
  ---                                                                                                                                         
  Why this matters in real life                                                                                                               
                                                                                                                                              
  In a neural network:                   
  - prices is replaced by millions of weights                                                                                                 
  - The forward pass is the model's prediction                                                                                                
  - The loss measures how wrong the prediction is
  - loss.backward() computes gradients for every single weight                                                                                
  - An optimizer uses those gradients to nudge each weight in the direction that lowers the loss                                              
  - Repeat millions of times → the network learns                                                                                             
                                                                                                                                              
  Your compute_price_deltas is the simplest possible version of that exact mechanism. Master this and you understand how every PyTorch model  
  trains.                                                                                                                                     
  ---     
"""
def compute_price_deltas(prices: torch.Tensor) -> torch.Tensor:
    # enable gradient tracking on prices
    prices.requires_grad_(True)
    # forward pass: compute deltas, square them and take the mean (builds the augograd graph)
    loss = (prices[1:] - prices[:-1]).pow(2).mean()
    # backward pass: compute gradients of loss w.r.t. prices
    loss.backward()
    # return the gradient tensor
    return prices.grad


"""

  How it mirrors the autograd version:
  - Same forward computation: deltas = prices[1:] - prices[:-1]
  - Loss is (1/N) * Σ deltas² → d(loss)/d(deltas[i]) = (2/N) * deltas[i]
  - Each delta depends on two prices with derivatives -1 and +1
"""


def compute_price_deltas_manual(prices: torch.Tensor) -> torch.Tensor:
    # detach to avoid any autograde tracking
    prices = prices.detach()

    # forward: compute deltas the same way as the autograd version
    deltas = prices[1:] - prices[:-1]
    N = len(deltas)
    # gradient tensor - same size as prices initialized all with zeroes
    grad = torch.zeros_like(prices)
    for i in range(N):
        grad[i] += -2 * deltas[i] / N  # contribution from the "current price" term
        grad[i + 1] += 2 * deltas[i] / N  # contribution from the "next price" term
    return grad
