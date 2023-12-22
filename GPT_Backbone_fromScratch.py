# GPT_Backbone_fromScratch.py  Software AnatomyNLM
# Copyright 2023 b<>com. All rights reserved.
# This software is licensed under the Apache License, Version 2.0.
# You may not use this file except in compliance with the license. 
# You may obtain a copy of the license at: 
# http://www.apache.org/licenses/LICENSE-2.0 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and 
# limitations under the License.


"""
Created on Wed Sep 27 07:46:20 2023

@author: Majd SALEH
"""

import keras_nlp
import numpy as np

def modelExplorer(model):
    # Input: model
    # Output: print model summary and model trainable parameters
    model.summary(expand_nested=True) # print model summary 
    variable_index=0
    for superlayer in model.layers: # print trainable variables
        print("------------------------")
        print("Layer name: ", superlayer.name)
        for tv in superlayer.trainable_variables:
            print(f'Variable {variable_index}: {tv.name:50} ===> Shape: {tv.shape}')
            variable_index+=1
            
def load_GPT2_Backbone_parameters(Pretrained_GPT2_Backbone):
    # Input : Pretrained_GPT2_Backbone 
    # Output: params_dict: trainable parameters dictionary

    # Hyper parameters 
    v=50257 # vocabulary size
    n_max=1024 # max sequence length
    d_e=768 # embedding dimension (model dimension)
    M=12 # numner of attention heads
    L=12 # number of transformer blocks 
    d_k=64 # key dimensions (same as query dimension)
    d_v=64 # value dimension
    d_f=3072 # feedforward dimension (internal dimension)
    # -------------------------------------------------------------------------
    # Weight matrices initialized by zeros
    E=np.zeros([d_e,v]) # Embeddings
    Lambda=np.zeros([d_e,n_max]) # Positional encoding
    # Transformer
    W_Q=np.zeros([L,M,d_e,d_k]) # Query kernel
    b_Q=np.zeros([L,M,d_k]) # Query bias
    W_K=np.zeros([L,M,d_e,d_k]) # Key kernel
    b_K=np.zeros([L,M,d_k]) # Key bias
    W_V=np.zeros([L,M,d_e,d_v]) # Value kernel
    b_V=np.zeros([L,M,d_v]) # Value bias
    W_O=np.zeros([L,M*d_v,d_e]) # MultiHeadAttention output kernel
    b_O=np.zeros([L,d_e]) # MultiHeadAttention output bias
    alpha_1=np.zeros([L,d_e]) # Layernorm 1 alpha
    beta_1=np.zeros([L,d_e]) # Layernorm 1 beta
    W_1=np.zeros([L,d_f,d_e]) # Feedforward kernel 1
    b_1=np.zeros([L,d_f]) # Feedforward bias 1
    W_2=np.zeros([L,d_e,d_f]) # Feedforward kernel 2
    b_2=np.zeros([L,d_e]) # Feedforward bias 2
    alpha_2=np.zeros([L,d_e]) # Layernorm 2 alpha
    beta_2=np.zeros([L,d_e]) # Layernorm 2 beta
    # Embedding layer norm
    alpha_e=np.zeros([d_e]) # Layernorm alpha_e
    beta_e=np.zeros([d_e])  # Layernorm beta_e
    # -------------------------------------------------------------------------
    # Extract the trainable parameters from the model and reshape them to match 
    # the notation used in the tutorial
    
    tv=model.trainable_variables # list of trainable parameters
    # the indices of parameters in the parameters list are obtained using the 
    # function modelExplorer()
    
    E=np.transpose(tv[0].numpy())
    Lambda=np.transpose(tv[1].numpy())
    alpha_e=tv[194].numpy()                       
    beta_e=tv[195].numpy()
    
    for i in range(L): # loop over the L transformer blocks and extract params
        tv_index=2+16*i
        W_Q[i]=tv[tv_index].numpy().swapaxes(0,1)
        
        tv_index+=1
        b_Q[i]=tv[tv_index].numpy()
        
        tv_index+=1
        W_K[i]=tv[tv_index].numpy().swapaxes(0,1)
        
        tv_index+=1
        b_K[i]=tv[tv_index].numpy()
        
        tv_index+=1
        W_V[i]=tv[tv_index].numpy().swapaxes(0,1)
        
        tv_index+=1
        b_V[i]=tv[tv_index].numpy()
        
        tv_index+=1    
        W_O_l=tv[tv_index].numpy()    
        W_O[i]=np.reshape(W_O_l, [M*d_v,d_e],"C")
        
        tv_index+=1
        b_O[i]=tv[tv_index].numpy()
        
        tv_index+=1
        alpha_1[i]=tv[tv_index].numpy()
        
        tv_index+=1
        beta_1[i]=tv[tv_index].numpy()
        
        tv_index+=1
        W_1[i]=np.transpose(tv[tv_index].numpy())
        
        tv_index+=1       
        b_1[i]=tv[tv_index].numpy()
        
        tv_index+=1
        W_2[i]=np.transpose(tv[tv_index].numpy())
        
        tv_index+=1       
        b_2[i]=tv[tv_index].numpy()
        
        tv_index+=1
        alpha_2[i]=tv[tv_index].numpy()
        
        tv_index+=1
        beta_2[i]=tv[tv_index].numpy()
        
    # package trainable parameters in one dictionary
    params_dict={"E":E,
                "Lambda":Lambda,
                "W_Q":W_Q,
                "b_Q":b_Q,
                "W_K":W_K,
                "b_K":b_K,
                "W_V":W_V,
                "b_V":b_V,
                "W_O":W_O,
                "b_O":b_O,
                "alpha_1":alpha_1,
                "beta_1":beta_1,
                "W_1":W_1,
                "b_1":b_1,
                "W_2":W_2,
                "b_2":b_2,
                "alpha_2":alpha_2,
                "beta_2":beta_2,
                "alpha_e":alpha_e,
                "beta_e":beta_e}
    
    return params_dict

def GPT2_Backbone_Tuto(Input_Sequence,GPT2_Backbone_Params):
    # GPT2_Backbone implementation using the tutorial equations
    # Inputs: 
    # Input_Sequence: a list containing the indices of tokens in the input sequence
    # GPT2_Backbone_Params : a dictionary of all trainable params of GP2_Backbone
    # Output:
    # HL: a (d_e x n) matrix representing the output of the GPT2_Backbone
    # Unpack the parameters dictionary
    E=GPT2_Backbone_Params["E"]
    Lambda=GPT2_Backbone_Params["Lambda"]
    W_Q=GPT2_Backbone_Params["W_Q"]
    b_Q=GPT2_Backbone_Params["b_Q"]
    W_K=GPT2_Backbone_Params["W_K"]
    b_K=GPT2_Backbone_Params["b_K"]
    W_V=GPT2_Backbone_Params["W_V"]
    b_V=GPT2_Backbone_Params["b_V"]
    W_O=GPT2_Backbone_Params["W_O"]
    b_O=GPT2_Backbone_Params["b_O"]
    alpha_1=GPT2_Backbone_Params["alpha_1"]
    beta_1=GPT2_Backbone_Params["beta_1"]
    W_1=GPT2_Backbone_Params["W_1"]
    b_1=GPT2_Backbone_Params["b_1"]
    W_2=GPT2_Backbone_Params["W_2"]
    b_2=GPT2_Backbone_Params["b_2"]
    alpha_2=GPT2_Backbone_Params["alpha_2"]
    beta_2=GPT2_Backbone_Params["beta_2"]
    alpha_e=GPT2_Backbone_Params["alpha_e"]
    beta_e=GPT2_Backbone_Params["beta_e"]
    
    # Hyper parameters 
    v=E.shape[1] # vocabulary size
    d_e=E.shape[0] # embedding dimension (model dimension)
    M=W_Q.shape[1] # numner of attention heads
    L=W_Q.shape[0] # number of transformer blocks 
    d_k=W_Q.shape[3] # key dimensions (same as query dimension)
    d_v=W_V.shape[3] # value dimension
    
    n=len(Input_Sequence) # sequence length
    
    # one-hot vectors
    Omega=np.zeros([v,n])
    
    for i in range(n):
        Omega[Input_Sequence[i],i]=1
    
    # Input embeddings
    X=np.matmul(E,Omega)
    
    # Corresponding positional encoding
    Lambda=Lambda[:,:n]
    
    # Input
    H0=X+Lambda
    
    # Mask S
    S_mask=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if j>i:
                S_mask[i,j]=-np.inf            
    
    # Transformer
    Hl=H0
    for l in range(L):
        residual=Hl
        # Layer norm
        mu = np.mean(Hl,axis=0)
        sigma= np.std(Hl,axis=0)        
        Hl=(Hl-mu)/sigma        
        Hl=Hl*np.transpose(np.tile(alpha_1[l],(n,1)))+np.transpose(np.tile(beta_1[l],(n,1)))
        
        # Equation 28: multi-head attention
        P=np.zeros([n,d_v,M])
        for m in range(M):
            Q=np.matmul(np.transpose(Hl),W_Q[l][m])
            Q=Q+b_Q[l][m]
            
            K=np.matmul(np.transpose(Hl),W_K[l][m])
            K=K+b_K[l][m]
            
            V=np.matmul(np.transpose(Hl),W_V[l][m])
            V=V+b_V[l][m]
            
            QKt=np.matmul(Q,np.transpose(K))
            
            X=QKt/np.sqrt(d_k) 
            Xm=S_mask+X    
            XmE=np.exp(Xm)
            X=XmE/np.transpose(np.tile(np.sum(XmE,axis=1),(n,1))) # softmax
        
            P[:,:,m]=np.matmul(X,V)
        
        P=np.reshape(P, [n,M*d_v],"F")
        
        A=np.transpose(np.matmul(P,W_O[l]))
        
        A=A+b_O[l].reshape([d_e,1])
        
        X=A+residual # resedual connection
        
        residual=X
        
        # Layer norm
        mu = np.mean(X,axis=0)
        sigma= np.std(X,axis=0)        
        X=(X-mu)/sigma        
        C=X*np.transpose(np.tile(alpha_2[l],(n,1)))+np.transpose(np.tile(beta_2[l],(n,1)))
    
        # Equation 37 : ffnn step
        D=np.zeros([d_e,n])
        for i in range(n):
            c_i=C[:,i]
            X1=np.matmul(W_1[l],c_i)+b_1[l]
            
            # gelu GELU(x)=0.5 x (1+tanh⁡(√(2/π)(x+0.044715 x^3)))
            X1=0.5*X1*(1+np.tanh(np.sqrt(2/np.pi)*(X1+0.044715*X1**3)))
    
            # 2nd layer
            D[:,i]=np.matmul(W_2[l],X1)+b_2[l]
        
        # output
        Hl=residual+D # residual connection
    
    HL=Hl
    # layer norm
    mu = np.mean(HL,axis=0) # mean
    sigma= np.std(HL,axis=0) # standard deviation
    HL=(HL-mu)/sigma # z-score
    HL=HL*np.transpose(np.tile(alpha_e,(n,1)))+np.transpose(np.tile(beta_e,(n,1)))
    
    return HL

if __name__ == '__main__':
    
    # input sequence
    Input_Sequence=[5338,318,100]
    
    # uncomment the following lines to test with random sequences
    # v=50257 # vocabulary size used by GPT2
    # n_max=1024 # max sequence length accepted by GPT2
    # n=3 # arbitrary sequence length
    
    # if n> n_max:
    #     n=n_max
    # Input_Sequence=np.random.randint(low=1, high=v, size=n)
    
    # Load pretrained gpt2 backbone (original KerasNLP implementation)
    model = keras_nlp.models.GPT2Backbone.from_preset("gpt2_base_en")
    
    # explore the model
    modelExplorer(model)
    
    # Evaluate GPT2Backbone on the Input_Sequence
    input_data = {
        "token_ids": np.array([Input_Sequence], dtype="int32"),
        "padding_mask": np.array([np.ones(len(Input_Sequence))]),
    }

    output_Original=np.transpose(model(input_data).numpy().squeeze())
    
    GPT2_Backbone_Params=load_GPT2_Backbone_parameters(model)
    
    # Evaluate our implementation of gpt2 backbone on the Input_Sequence
    output_Tutorial=GPT2_Backbone_Tuto(Input_Sequence,GPT2_Backbone_Params)
    
    print("-"*60)
    print("Result matrix of the original implementation of GP2_Backbone:")
    print(output_Original)
    
    print("-"*60)
    print("Result matrix of our implementation of GP2_Backbone (tutorial equations):")
    print(output_Tutorial)
    print("-"*60)