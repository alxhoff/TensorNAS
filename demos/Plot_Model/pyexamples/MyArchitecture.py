import sys
import os
import subprocess
sys.path.append('../')
from ..pycore import tikzeng, blocks


class Plot_Model():
    # defined your arch
    layer_list=[]
    def __init__(self,layer_name,args_items):
        self.args_items=args_items
        self.layer_name=layer_name




    def Plot_Layer(layer_names,args_items):
            arch = [
                tikzeng.to_head('..'),
                tikzeng.to_cor(),
                tikzeng.to_begin()
            ]
            # ['Conv2D', 'MaxPool2D', 'Flatten', 'Dropout', 'OutputDense', 'Conv2D', 'MaxPool2D', 'Flatten', 'Dropout', 'OutputDense']

            for layer_name,layer_args in zip(layer_names,args_items):
                #layer_args:dict
                if layer_name == 'Conv2D':
                    for key, value in layer_args.items():
                        if key.name == "KERNEL_SIZE":
                            s_filer=value
                        if key.name =="FILTERS":
                            n_filer=value
                    arch.append(tikzeng.to_Conv(layer_name,offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" "))

                if layer_name == 'MaxPool2D':
                    for key,value in layer_args.items():
                        if key.name =="POOL_SIZE":
                            pool_size=value
                    arch.append(tikzeng.to_Pool(layer_name, pool_size, offset="(0,0,0)", to="(cr1-east)",
                            width=1, height=35, depth=35, opacity=0.5))


                if layer_name == 'HiddenDense':
                    for key, value in layer_args.items():
                        if key.name == "UNITS":
                            units = value
                    arch.append(tikzeng.to_FullyConnected(layer_name,units, offset="(1.25,0,0)",
                                              to="(fl-east)", width=1, height=1, depth=40,caption="fc1\ndr"))

                if layer_name == 'OutputDense':
                    for key, value in layer_args.items():
                        if key.name == "UNITS":
                            units = value
                    arch.append(tikzeng.to_SoftMax(layer_name,units, offset="(1.25,0,0)", to="(fc3-east)",
                               width=1, height=1, depth=10,
                               caption="SIGMOID", opacity=1.0))

            namefile = str(os.path.splitext(os.path.basename(__file__))[0])
            #print(os.path.splitext("/path/to/some/file.txt")[0])
            tikzeng.to_generate(arch, namefile + '.tex')
           # os.system("sh ../tikzmake.sh MyArchitecture ../../../../demos/MyArchitecture ")
            #os.system("mv MyArchitecture ../../../../demos/MyArchitecture ")
# sh ../tikzmake.sh pyexamples/MyArchitecture ../../DemoClassification






