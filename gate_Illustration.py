# gate_Illustration.py – Software AnatomyNLM
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
Created on Mon Oct  9 09:01:08 2023

@author: Majd SALEH
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# plot configurations
plt.style.use('default')
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica",
  "font.size":12
})

# the sigmoid function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    # vector dimension
    d_e=20
    
    # vector used for gating
    # let's generate a random vector sampled from a normal distribution with mean 
    # mu and standard deviation sigma
    mu=2
    sigma=30
    
    x=mu+sigma*np.random.randn(d_e)
    
    # apply sigmoid to obtain a semi-binary mask
    mask=sigmoid(x)
    
    # vector to be gated (generated randomly from a uniform distribution over [0, 1))
    v= np.random.rand(d_e)
    
    # Gating (elementwise multiplication of v and the semi-binary mask)
    v_gated=v*mask
    
    
    # padding (to better visualize vector components)
    mask=np.insert(mask,0,mask[0])
    v=np.insert(v,0,v[0])
    v_gated=np.insert(v_gated,0,v_gated[0])
    
    mask=np.insert(mask,len(mask),mask[-1])
    v=np.insert(v,len(v),v[-1])
    v_gated=np.insert(v_gated,len(v_gated),v_gated[-1])
    
    # plot and save
    x_axis=np.array(list(range(d_e+2)))
    
    f, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(21,11))
    f.set_dpi(150)
    # vector to be gated
    ax1.step(x_axis,v,'g-',label='vector to be gated',where='mid',linewidth=3)
    ax1.set_title('a)',fontsize = 20)
    ax1.set_ylabel('component value',fontsize = 20)
    ax1.legend(loc="upper right",fontsize=20)
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(True, which='minor')
    ax1.set_axisbelow(True)
    ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.set_xticks(x_axis)
    ax1.set_xticklabels(x_axis)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlim([0.5, d_e+0.5])
    
    # semi-binary mask
    ax2.step(x_axis,mask,'r-',label='semi-binary mask',where='mid',linewidth=3)
    ax2.set_title('b)',fontsize = 20)
    ax2.set_ylabel('component value',fontsize = 20)
    ax2.legend(loc="upper right",fontsize=20)
    ax2.yaxis.grid(True)
    ax2.xaxis.grid(True, which='minor')
    ax2.set_axisbelow(True)
    ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2.set_xticks(x_axis)
    ax2.set_xticklabels(x_axis)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlim([0.5, d_e+0.5])
    
    # gated vector
    ax3.step(x_axis,v_gated,'b-',label='gated vector',where='mid',linewidth=3)
    ax3.set_title('c)',fontsize = 20)
    ax3.set_ylabel('component value',fontsize = 20)
    ax3.set_xlabel('component index',fontsize = 20)
    ax3.legend(loc="upper right",fontsize=20)
    ax3.yaxis.grid(True)
    ax3.xaxis.grid(True, which='minor')
    ax3.set_axisbelow(True)
    ax3.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax3.yaxis.set_major_locator(MultipleLocator(0.2))
    ax3.set_xticks(x_axis)
    ax3.set_xticklabels(x_axis)
    ax3.tick_params(axis='both', which='major', labelsize=20)
    ax3.set_ylim([-0.05, 1.05])
    ax3.set_xlim([0.5, d_e+0.5])
    
    # save figure
    plt.savefig("gate_concept.pdf", dpi='figure')
    # plt.savefig("gate_concept.eps", format='eps', dpi='figure')
    # plt.savefig("gate_concept.svg", format="svg",transparent=True)
    
    # show figure
    plt.show()
    
    x=np.array(list(range(-100,100,1)))/10
    y=sigmoid(x)
    plt.plot(x,y)
    
    f, (ax1) = plt.subplots(1, 1,figsize=(7,3))
    f.set_dpi(150)
    # vector to be gated
    ax1.plot(x,y,'b-',linewidth=2)
    ax1.set_title('sigmiod',fontsize = 20)
    ax1.set_xlabel('$x$',fontsize = 20)
    ax1.set_ylabel('$y=\mathrm\sigma(x)$',fontsize = 20)
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(True)
    ax1.set_axisbelow(True)    
    ax1.set_ylim([0, 1])
    ax1.set_xlim([-10, 10])
    
    plt.savefig("simoid.svg", dpi='figure')