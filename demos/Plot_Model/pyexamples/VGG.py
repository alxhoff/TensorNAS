import sys
sys.path.append('../')
from tensornas.core.Plot_Model.pycore.tikzeng import *


c
# defined your arch
arch = [
        to_head( '..' ),
        to_cor(),
        to_begin(),
# input
#to_input('input.png', width=25.6, height=5.0, name='input'),

# conv1
to_ConvConvRelu(name='cr1', s_filer=256, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)", width=(2, 2), height=40,
                depth=40, caption="conv1"),
to_Pool(name="p1", offset="(0,0,0)", to="(cr1-east)", width=1, height=35, depth=35, opacity=0.5),

# conv2
to_ConvConvRelu(name='cr2', s_filer=128, n_filer=(128, 128), offset="(2,0,0)", to="(p1-east)", width=(4, 4), height=35,
                depth=35, caption="conv2"),
to_Pool(name="p2", offset="(0,0,0)", to="(cr2-east)", width=1, height=30, depth=30, opacity=0.5),

# conv3
to_ConvConvRelu(name='cr3', s_filer=64, n_filer=("256", "256", "256"), offset="(2,0,0)", to="(p2-east)",
                width=(4, 4, 4), height=30, depth=30, caption="conv3"),
to_Pool(name="p3", offset="(0,0,0)", to="(cr3-east)", width=1, height=23, depth=23, opacity=0.5),

# conv4
to_ConvConvRelu(name='cr4', s_filer=32, n_filer=("512", "512", "512"), offset="(2,0,0)", to="(p3-east)",
                width=(4, 4, 4), height=23, depth=23, caption="conv4"),
to_Pool(name="p4", offset="(0,0,0)", to="(cr4-east)", width=1, height=15, depth=15, opacity=0.5),

# conv5
to_ConvConvRelu(name='cr5', s_filer=16, n_filer=("512", "512", "512"), offset="(2,0,0)", to="(p4-east)",
                width=(4, 4, 4), height=15, depth=15, caption="conv5"),
to_Pool(name="p5", offset="(0,0,0)", to="(cr5-east)", width=1, height=10, depth=10, opacity=0.5),

# flatten
to_FullyConnected(name="fl", s_filer=4096, offset="(1.25,0,0)", to="(p5-east)", width=1, height=1, depth=20,
                  caption="fl"),

# fc1
to_FullyConnected(name="fc1", s_filer=8192, offset="(1.25,0,0)", to="(fl-east)", width=1, height=1, depth=40,
                  caption="fc1\ndr"),

# fc2
to_FullyConnected(name="fc2", s_filer=8192, offset="(1.25,0,0)", to="(fc1-east)", width=1, height=1, depth=40,
                  caption="fc2\ndr"),

# fc3
to_FullyConnected(name="fc3", s_filer=8192, offset="(1.25,0,0)", to="(fc2-east)", width=1, height=1, depth=40,
                  caption="fc3\ndr"),

#to_input('output.png', width=25.6, height=5.0, name='output', to='(30, 0, 0)'),

# sigmoid
to_SoftMax(name="sg", n_filer="256", offset="(1.25,0,0)", to="(fc3-east)", width=1, height=1, depth=10,
           caption="SIGMOID", opacity=1.0),

# connections
to_connection("p1", "cr2"),
to_connection("p2", "cr3"),
to_connection("p3", "cr4"),
to_connection("p4", "cr5"),
to_connection("p5", "fl"),
to_connection("fl", "fc1"),
to_connection("fc1", "fc2"),
to_connection("fc2", "fc3"),
to_connection("fc3", "sg"),
to_connection("sg", "++(1.5,0,0)", to_dir=''),
to_end()
        ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()